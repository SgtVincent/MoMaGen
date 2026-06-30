"""
A collection of classes used to represent waypoints and trajectories.
"""
import json
import math
import os
import time
import traceback
import numpy as np
import copy
from copy import deepcopy

import momagen.utils.pose_utils as PoseUtils

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import (
    CuRoboEmbodimentSelection,
    default_embodiment_variant,
    is_default_embodiment,
)
from omnigibson.controllers import ControlType, HolonomicBaseJointController, JointController
import omnigibson.lazy as lazy
import torch as th
import omnigibson as og
from omnigibson import object_states
from omnigibson.robots.r1 import R1
from omnigibson.robots.r1pro import R1Pro
from omnigibson.robots.tiago import Tiago
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.usd_utils import GripperRigidContactAPI

from scipy.spatial.transform import Rotation as R


def _compute_trajectories_with_paths(cmg, **kwargs):
    """Normalize CuRobo compute_trajectories return values across OG versions."""
    result = cmg.compute_trajectories(**kwargs)
    if isinstance(result, tuple):
        return result

    mp_results = result
    traj_paths = []
    for mp_result in mp_results:
        if getattr(mp_result, "interpolated_plan", None) is None:
            traj_paths.extend([None] * int(mp_result.success.shape[0]))
        else:
            traj_paths.extend(mp_result.get_paths())
    return mp_results, traj_paths


def _mp_status_value(mp_result):
    """Return a string CuRobo status, tolerating admission-filtered results with status=None."""
    status_obj = getattr(mp_result, "status", None)
    return str(getattr(status_obj, "value", status_obj))


def _attached_payload_options(attached_obj, base_options=None, ee_pose_by_link=None, remove_from_world=False):
    """Build CuRobo attach options for objects currently carried by an end-effector."""
    if not attached_obj:
        return base_options

    options = copy.deepcopy(base_options) if base_options is not None else {}
    for link_name in attached_obj.keys():
        link_options = dict(options.get(link_name, {}))
        if remove_from_world:
            link_options.setdefault("remove_obstacles_from_world_config", True)
        if ee_pose_by_link is not None and link_name in ee_pose_by_link:
            link_options["ee_pose"] = ee_pose_by_link[link_name]
        options[link_name] = link_options
    return options


def _attached_payload_link_pair_collision_pairs(robot, attached_obj):
    """Build manipulator-vs-attached-payload CuRobo link pairs for selective collision validation."""
    if not attached_obj:
        return []

    pairs = []
    for holder_eef_link in attached_obj.keys():
        holder_arm = None
        for arm_name, eef_link_name in robot.eef_link_names.items():
            if eef_link_name == holder_eef_link:
                holder_arm = arm_name
                break
        if holder_arm is None:
            continue

        attached_link_name = robot.curobo_attached_object_link_names.get(holder_eef_link)
        if attached_link_name is None:
            continue

        for arm_name in robot.eef_link_names.keys():
            if arm_name == holder_arm:
                continue
            active_arm_links = [
                f"{arm_name}_arm_link{i}" for i in range(1, 8)
            ] + [
                f"{arm_name}_gripper_link",
                f"{arm_name}_gripper_finger_link1",
                f"{arm_name}_gripper_finger_link2",
            ]
            pairs.append((active_arm_links, [attached_link_name]))
    return pairs


def _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01):
    """Add joint-space waypoints across supported OmniGibson primitive APIs."""
    if len(q_traj) <= 1:
        return q_traj

    # Older MoMaGen expected this helper on StarterSemanticActionPrimitives, while
    # current BEHAVIOR-1K / OmniGibson exposes the public helper on CuRoboMotionGenerator.
    primitive_interp = getattr(getattr(env, "primitive", None), "_add_linearly_interpolated_waypoints", None)
    if primitive_interp is not None:
        return th.stack(primitive_interp(plan=q_traj, max_inter_dist=max_inter_dist))

    # Current BEHAVIOR-1K exposes CuRoboMotionGenerator.add_linearly_interpolated_waypoints,
    # but that implementation calls a torchscript multi_dim_linspace helper that can fail
    # under Isaac / CUDA with "Global alloc not supported yet". Keep this small Python
    # equivalent local to MoMaGen so generation does not depend on that JIT path.
    interpolated_plan = []
    for i in range(len(q_traj) - 1):
        max_diff = (q_traj[i + 1] - q_traj[i]).abs().max()
        num_intervals = max(1, int(np.ceil(max_diff.item() / max_inter_dist)))
        steps = th.linspace(
            0,
            1,
            num_intervals + 1,
            dtype=q_traj.dtype,
            device=q_traj.device,
        )[:-1]
        steps = steps.reshape([num_intervals] + [1] * q_traj[i].dim())
        interpolated_plan.append(q_traj[i] + steps * (q_traj[i + 1] - q_traj[i]))

    interpolated_plan.append(q_traj[-1].unsqueeze(0))
    return th.cat(interpolated_plan, dim=0)


def _safe_index_tensor(idx, *, device=None):
    if idx is None:
        return None
    if isinstance(idx, dict):
        tensors = []
        for value in idx.values():
            tensor = _safe_index_tensor(value, device=device)
            if tensor is not None and tensor.numel() > 0:
                tensors.append(tensor)
        if not tensors:
            return None
        return th.cat(tensors).unique(sorted=True)
    if isinstance(idx, slice):
        return idx
    if isinstance(idx, th.Tensor):
        return idx.to(device=device) if device is not None else idx
    return th.as_tensor(idx, dtype=th.long, device=device)


def _debug_to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=float)


def maybe_apply_phase_routing_target_precontact(target_pose, *, env, ref_obj, object_ref=None, phase_type=None, phase_logs=None):
    """Optionally adjust phase-routing targets around the referenced object."""
    if not bool(int(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "0") or 0)):
        return target_pose, None

    phase = int(getattr(env, "execution_phase_ind", 0))
    min_phase = int(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_MIN_PHASE", "0") or 0)
    max_phase = int(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_MAX_PHASE", "999999") or 999999)
    record = {
        "enabled": True,
        "phase": phase,
        "phase_type": phase_type,
        "object_ref": object_ref,
        "min_phase": min_phase,
        "max_phase": max_phase,
    }

    def _record_and_return(adjusted_pose, reason=None):
        if reason is not None:
            record.setdefault("applied", False)
            record["reason"] = reason
        if phase_logs is not None:
            phase_logs.setdefault(phase, {}).setdefault("phase_routing_target_precontact", []).append(record)
        print("[MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT] " + json.dumps(record, default=str), flush=True)
        return adjusted_pose, record

    if not (min_phase <= phase <= max_phase):
        return _record_and_return(target_pose, "phase_out_of_range")
    if ref_obj is None:
        return _record_and_return(target_pose, "missing_ref_obj")
    if not isinstance(target_pose, dict):
        return _record_and_return(target_pose, "target_pose_not_dict")

    ref_mode = (os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_REF_MODE", "marker") or "marker").strip().lower()
    ref_source = "object"
    ref_quat_raw = None
    try:
        if ref_mode == "marker" and object_states.ToggledOn in getattr(ref_obj, "states", {}):
            marker = getattr(ref_obj.states[object_states.ToggledOn], "visual_marker", None)
            if marker is not None:
                ref_pos_raw, ref_quat_raw = marker.get_position_orientation()
                ref_source = "marker"
            else:
                ref_pos_raw, ref_quat_raw = ref_obj.get_position_orientation()
        else:
            ref_pos_raw, ref_quat_raw = ref_obj.get_position_orientation()
    except Exception as e:
        return _record_and_return(target_pose, f"ref_pose_error:{type(e).__name__}: {e}")

    distance = float(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_DISTANCE", "0.0") or 0.0)
    z_offset = float(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_Z", "0.0") or 0.0)
    approach_vector_raw = (os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "") or "").strip()
    approach_vector_frame = (
        os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR_FRAME", "world") or "world"
    ).strip().lower()
    finger_link_goal_enabled = bool(
        int(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "0") or 0)
    )
    finger_link_goal_distance = float(
        os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_DISTANCE", "0.0") or 0.0
    )
    finger_link_goal_z = float(os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_Z", "0.0") or 0.0)
    finger_link_marker_local_offset_raw = (
        os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_MARKER_LOCAL_OFFSET", "") or ""
    ).strip()
    finger_link_marker_local_offset = None
    if finger_link_marker_local_offset_raw:
        finger_link_marker_local_offset_values = [
            float(value.strip())
            for value in finger_link_marker_local_offset_raw.split(",")
            if value.strip()
        ]
        if len(finger_link_marker_local_offset_values) != 3:
            return _record_and_return(
                target_pose,
                "invalid_finger_link_marker_local_offset",
            )
        finger_link_marker_local_offset = finger_link_marker_local_offset_values
    force_finger_link = (
        os.environ.get(
            "MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK",
            os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_FORCE_FINGER_LINK", ""),
        )
        or ""
    ).strip()
    active_arms_raw = (os.environ.get("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "auto") or "auto").strip()
    if active_arms_raw.lower() == "auto":
        active_arms = set(target_pose.keys())
    else:
        active_arms = {value.strip() for value in active_arms_raw.split(",") if value.strip()}
    if not active_arms:
        return _record_and_return(target_pose, "no_active_arms")

    adjusted_pose = dict(target_pose)
    ref_pos_np = _debug_to_np(ref_pos_raw)
    if ref_pos_np is None or ref_pos_np.shape[0] < 3 or not np.isfinite(ref_pos_np[:3]).all():
        return _record_and_return(target_pose, "invalid_ref_pos")
    approach_vector_values = None
    if approach_vector_raw:
        approach_vector_values = [float(value.strip()) for value in approach_vector_raw.split(",") if value.strip()]
        if len(approach_vector_values) != 3:
            return _record_and_return(target_pose, "invalid_approach_vector")
        if approach_vector_frame not in {"world", "marker", "ref"}:
            return _record_and_return(target_pose, "invalid_approach_vector_frame")
        if approach_vector_frame in {"marker", "ref"} and ref_quat_raw is None:
            return _record_and_return(target_pose, "missing_ref_quat_for_approach_vector")

    arm_records = []
    any_applied = False
    eps = 1e-6
    robot = getattr(env, "robot", None)
    if robot is None:
        robot = getattr(getattr(env, "env", None), "robots", [None])[0]

    for arm_name, pose in list(target_pose.items()):
        if arm_name not in active_arms:
            arm_records.append({"arm": arm_name, "applied": False, "reason": "arm_not_active"})
            continue
        try:
            pos, quat = pose
            pos_tensor = th.as_tensor(pos)
            original_dtype = pos_tensor.dtype
            original_device = pos_tensor.device
            if not th.is_floating_point(pos_tensor):
                pos_tensor = pos_tensor.float()
            ref_pos = th.as_tensor(ref_pos_np[:3], dtype=pos_tensor.dtype, device=pos_tensor.device)
            marker_local_offset_world = th.zeros(3, dtype=pos_tensor.dtype, device=pos_tensor.device)
            if finger_link_marker_local_offset is not None:
                if ref_quat_raw is None:
                    arm_records.append(
                        {
                            "arm": arm_name,
                            "applied": False,
                            "reason": "missing_ref_quat_for_marker_local_offset",
                        }
                    )
                    continue
                marker_quat = th.as_tensor(ref_quat_raw, dtype=pos_tensor.dtype, device=pos_tensor.device)
                marker_rot = th.as_tensor(T.quat2mat(marker_quat), dtype=pos_tensor.dtype, device=pos_tensor.device)
                marker_local_offset = th.tensor(
                    finger_link_marker_local_offset,
                    dtype=pos_tensor.dtype,
                    device=pos_tensor.device,
                )
                marker_local_offset_world = marker_rot @ marker_local_offset
            direction = pos_tensor - ref_pos
            norm = th.linalg.norm(direction, dim=-1, keepdim=True)
            if bool(th.any(norm <= eps)):
                arm_records.append({"arm": arm_name, "applied": False, "reason": "zero_ref_to_target_direction"})
                continue
            unit_direction = direction / norm.clamp_min(eps)
            approach_vector_world = None
            if approach_vector_values is not None:
                approach_vector = th.tensor(
                    approach_vector_values,
                    dtype=pos_tensor.dtype,
                    device=pos_tensor.device,
                )
                if approach_vector_frame in {"marker", "ref"}:
                    ref_quat = th.as_tensor(ref_quat_raw, dtype=pos_tensor.dtype, device=pos_tensor.device)
                    ref_rot = th.as_tensor(T.quat2mat(ref_quat), dtype=pos_tensor.dtype, device=pos_tensor.device)
                    approach_vector = ref_rot @ approach_vector
                approach_norm = th.linalg.norm(approach_vector)
                if bool(approach_norm <= eps):
                    arm_records.append({"arm": arm_name, "applied": False, "reason": "zero_approach_vector"})
                    continue
                unit_direction = approach_vector / approach_norm.clamp_min(eps)
                approach_vector_world = unit_direction
            z_delta = th.zeros_like(pos_tensor)
            z_delta[..., 2] = z_offset
            adjusted_pos = pos_tensor + distance * unit_direction + z_delta
            if hasattr(pos, "detach"):
                adjusted_pos = adjusted_pos.to(dtype=original_dtype, device=original_device)
            elif isinstance(pos, np.ndarray):
                adjusted_pos = adjusted_pos.detach().cpu().numpy().astype(pos.dtype, copy=False)
            adjusted_pose[arm_name] = (adjusted_pos, quat)
            finger_link_goal_record = None
            if finger_link_goal_enabled:
                finger_links = getattr(robot, "finger_links", {}).get(arm_name, []) if robot is not None else []
                finger_records = []
                for finger_link in finger_links:
                    finger_name = getattr(finger_link, "name", str(finger_link))
                    if force_finger_link and force_finger_link != finger_name and force_finger_link not in finger_name:
                        continue
                    finger_body_name = getattr(finger_link, "body_name", None)
                    if finger_body_name is None:
                        finger_body_name = str(finger_name).split(":")[-1]
                    if robot is not None and finger_body_name not in getattr(robot, "links", {}):
                        continue
                    finger_records.append((finger_link, finger_body_name))
                if finger_records:
                    finger_link, finger_body_name = finger_records[0]
                    finger_quat = th.as_tensor(
                        finger_link.get_position_orientation()[1],
                        dtype=pos_tensor.dtype,
                        device=pos_tensor.device,
                    )
                    finger_target_pos = ref_pos + marker_local_offset_world + finger_link_goal_distance * unit_direction + th.tensor(
                        [0.0, 0.0, finger_link_goal_z],
                        dtype=pos_tensor.dtype,
                        device=pos_tensor.device,
                    )
                    if pos_tensor.ndim == 2:
                        finger_target_pos = finger_target_pos.repeat(pos_tensor.shape[0], 1)
                        finger_quat = finger_quat.repeat(pos_tensor.shape[0], 1)
                    if hasattr(pos, "detach"):
                        finger_target_pos = finger_target_pos.to(dtype=original_dtype, device=original_device)
                        finger_quat = finger_quat.to(dtype=quat.dtype, device=quat.device) if hasattr(quat, "dtype") else finger_quat
                    elif isinstance(pos, np.ndarray):
                        finger_target_pos = finger_target_pos.detach().cpu().numpy().astype(pos.dtype, copy=False)
                    adjusted_pose[finger_body_name] = (finger_target_pos, finger_quat)
                    finger_link_goal_record = {
                        "enabled": True,
                        "applied": True,
                        "link": finger_body_name,
                        "distance": finger_link_goal_distance,
                        "marker_local_offset": finger_link_marker_local_offset,
                        "marker_local_offset_world": _debug_to_np(marker_local_offset_world).tolist(),
                        "approach_vector_world": _debug_to_np(approach_vector_world).tolist()
                        if approach_vector_world is not None
                        else None,
                        "target_pos": _debug_to_np(finger_target_pos).tolist(),
                        "target_quat": _debug_to_np(finger_quat).tolist(),
                    }
                else:
                    finger_link_goal_record = {
                        "enabled": True,
                        "applied": False,
                        "reason": "missing_matching_finger_link",
                        "force_finger_link": force_finger_link,
                    }
            any_applied = True
            arm_record = {
                "arm": arm_name,
                "applied": True,
                "original_pos": _debug_to_np(pos).tolist(),
                "adjusted_pos": _debug_to_np(adjusted_pos).tolist(),
                "delta_norm": float(th.linalg.norm(adjusted_pos - pos_tensor).item()),
            }
            if approach_vector_world is not None:
                arm_record["approach_vector_world"] = _debug_to_np(approach_vector_world).tolist()
            if finger_link_goal_record is not None:
                arm_record["finger_link_goal"] = finger_link_goal_record
            arm_records.append(arm_record)
        except Exception as e:
            arm_records.append({"arm": arm_name, "applied": False, "reason": f"{type(e).__name__}: {e}"})

    record.update(
        {
            "applied": bool(any_applied),
            "ref_mode": ref_mode,
            "ref_source": ref_source,
            "ref_pos": ref_pos_np[:3].tolist(),
            "distance": distance,
            "z_offset": z_offset,
            "approach_vector": approach_vector_values,
            "approach_vector_frame": approach_vector_frame,
            "finger_link_goal_enabled": finger_link_goal_enabled,
            "finger_link_goal_distance": finger_link_goal_distance,
            "finger_link_goal_z": finger_link_goal_z,
            "active_arms": sorted(active_arms),
            "arms": arm_records,
        }
    )
    if not any_applied:
        record["reason"] = "no_arm_adjusted"
    return _record_and_return(adjusted_pose)


def select_phase_routing_nav_eef_pose(eef_pose, selected_nav_arm):
    """Select nav EEF targets while preserving explicit robot-link targets."""
    if selected_nav_arm == "both":
        return eef_pose
    if selected_nav_arm not in {"left", "right"}:
        raise ValueError(f"Unsupported selected_nav_arm {selected_nav_arm!r}")
    selected_pose = {selected_nav_arm: eef_pose[selected_nav_arm]}
    selected_pose.update({key: value for key, value in eef_pose.items() if key not in {"left", "right"}})
    return selected_pose


def _joint_error_summary_by_group(robot, current_q, target_q):
    """Summarize joint tracking error by robot control group."""
    current_q = th.as_tensor(current_q).detach().cpu().float()
    target_q = th.as_tensor(target_q).detach().cpu().float()
    diff = (current_q - target_q).abs()

    def _group(name, idx):
        idx = _safe_index_tensor(idx)
        if idx is None:
            return None
        if isinstance(idx, slice):
            idx = th.arange(current_q.shape[0])[idx]
        idx = idx.detach().cpu().long()
        if idx.numel() == 0:
            return None
        vals = diff[idx]
        argmax_local = int(th.argmax(vals).item())
        joint_idx = int(idx[argmax_local].item())
        joint_names = getattr(robot, "dof_names_ordered", None)
        return {
            "max_abs": float(vals.max().item()),
            "mean_abs": float(vals.mean().item()),
            "argmax_joint_idx": joint_idx,
            "argmax_joint_name": None
            if joint_names is None or joint_idx >= len(joint_names)
            else str(joint_names[joint_idx]),
            "current_at_argmax": float(current_q[joint_idx].item()),
            "target_at_argmax": float(target_q[joint_idx].item()),
        }

    groups = {}
    for attr_name in ("base_control_idx", "base_idx", "trunk_control_idx"):
        if hasattr(robot, attr_name):
            record = _group(attr_name, getattr(robot, attr_name))
            if record is not None:
                groups[attr_name] = record
    for arm_name in ("left", "right"):
        for attr_name, prefix in (
            ("arm_control_idx", "arm"),
            ("gripper_control_idx", "gripper"),
        ):
            idx_by_arm = getattr(robot, attr_name, None)
            if isinstance(idx_by_arm, dict) and arm_name in idx_by_arm:
                record = _group(f"{prefix}_{arm_name}", idx_by_arm[arm_name])
                if record is not None:
                    groups[f"{prefix}_{arm_name}"] = record
    return groups


def _joint_path_quality_by_group(robot, q_path):
    """Summarize whole-body joint path length for candidate ranking."""
    q_path = th.as_tensor(q_path).detach().cpu().float()
    records = {}

    def _idx_tensor(idx):
        idx = _safe_index_tensor(idx)
        if idx is None:
            return None
        if isinstance(idx, slice):
            idx = th.arange(q_path.shape[1])[idx]
        return idx.detach().cpu().long()

    def _joint_names(idx):
        names = getattr(robot, "dof_names_ordered", None)
        if names is None:
            return None
        return [str(names[int(i)]) if int(i) < len(names) else str(int(i)) for i in idx.tolist()]

    def _path_metric(name, idx, *, yaw=False):
        idx = _idx_tensor(idx)
        if idx is None or idx.numel() == 0 or q_path.shape[0] <= 1:
            return None
        values = q_path[:, idx]
        if yaw and values.shape[1] >= 3:
            xy_step = th.linalg.norm(values[1:, :2] - values[:-1, :2], dim=-1)
            yaw_step = th.as_tensor(
                [abs(float(wrap_angle(v))) for v in (values[1:, 2] - values[:-1, 2])],
                dtype=values.dtype,
            )
            return {
                "path_m": float(xy_step.sum().item()),
                "net_m": float(th.linalg.norm(values[-1, :2] - values[0, :2]).item()),
                "path_rad": float(yaw_step.sum().item()),
                "net_rad": float(abs(wrap_angle(values[-1, 2] - values[0, 2]))),
                "max_step_m": float(xy_step.max().item()) if xy_step.numel() else 0.0,
                "max_step_rad": float(yaw_step.max().item()) if yaw_step.numel() else 0.0,
                "joint_names": _joint_names(idx),
            }
        step = th.linalg.norm(values[1:] - values[:-1], dim=-1)
        net = th.linalg.norm(values[-1] - values[0])
        return {
            "path": float(step.sum().item()),
            "net": float(net.item()),
            "path_to_net_ratio": float(step.sum().item() / max(float(net.item()), 1e-9)),
            "max_step": float(step.max().item()) if step.numel() else 0.0,
            "joint_names": _joint_names(idx),
        }

    if hasattr(robot, "base_control_idx"):
        metric = _path_metric("base", getattr(robot, "base_control_idx"), yaw=True)
        if metric is not None:
            records["base"] = metric
    if hasattr(robot, "trunk_control_idx"):
        metric = _path_metric("trunk", getattr(robot, "trunk_control_idx"))
        if metric is not None:
            records["trunk"] = metric
    arm_control_idx = getattr(robot, "arm_control_idx", None)
    if isinstance(arm_control_idx, dict):
        for arm_name in ("left", "right"):
            if arm_name in arm_control_idx:
                metric = _path_metric(f"arm_{arm_name}", arm_control_idx[arm_name])
                if metric is not None:
                    records[f"arm_{arm_name}"] = metric
    return records


def _action_limit_summary_by_controller(robot, action):
    """Summarize whether an action lies outside each controller's command limits."""
    action_np = np.asarray(_debug_to_np(action), dtype=float)
    records = {}
    for controller_name, action_idx in getattr(robot, "controller_action_idx", {}).items():
        controller = getattr(robot, "controllers", {}).get(controller_name)
        if controller is None:
            continue
        try:
            idx = _safe_index_tensor(action_idx)
            if isinstance(idx, slice):
                values = action_np[idx]
            else:
                values = action_np[idx.detach().cpu().long().numpy()]
            values = np.asarray(values, dtype=float)
            record = {
                "min": float(np.nanmin(values)) if values.size else None,
                "max": float(np.nanmax(values)) if values.size else None,
                "outside_action_space_count": 0,
            }
            limits = getattr(controller, "command_input_limits", None)
            if limits is not None:
                low = np.asarray(_debug_to_np(limits[0]), dtype=float)
                high = np.asarray(_debug_to_np(limits[1]), dtype=float)
                below = values < low
                above = values > high
                record.update(
                    {
                        "low_min": float(np.nanmin(low)) if low.size else None,
                        "high_max": float(np.nanmax(high)) if high.size else None,
                        "outside_action_space_count": int(np.count_nonzero(below | above)),
                        "max_below_limit": float(np.nanmax(np.where(below, low - values, 0.0)))
                        if values.size
                        else 0.0,
                        "max_above_limit": float(np.nanmax(np.where(above, values - high, 0.0)))
                        if values.size
                        else 0.0,
                    }
                )
            records[controller_name] = record
        except Exception as exc:
            records[controller_name] = {"error": f"{type(exc).__name__}: {exc}"}
    return records


def _joint_trajectory_point_to_action(robot, q):
    """Convert a planned joint point to action while tolerating non-joint gripper controllers."""
    try:
        return robot.q_to_action(q)
    except AssertionError as e:
        # Current BEHAVIOR-1K R1Pro runtime controllers can differ from the strict
        # HolonomicBaseRobot.q_to_action expectation in two ways:
        #   1. grippers may use non-JointController controllers, and
        #   2. arm JointControllers may use delta position commands.
        # Both are handled by the compatibility path below.  Other assertion
        # failures should still surface normally.
        if "Controller [gripper_" not in str(e) and "use_delta_commands=False" not in str(e):
            raise

    # Current BEHAVIOR-1K R1Pro configs may use non-JointController gripper controllers,
    # while MoMaGen overwrites gripper action entries from source waypoints immediately
    # after q_to_action. Reconstruct the non-gripper joint-controller action here.
    action = th.zeros(robot.action_dim, dtype=q.dtype, device=q.device)
    cur_joint_pos = robot.get_joint_positions().to(device=q.device, dtype=q.dtype)
    body_pose = robot.get_position_orientation()
    body_pos = body_pose[0].to(device=q.device, dtype=q.dtype)
    body_quat = body_pose[1].to(device=q.device, dtype=q.dtype)

    for name, controller in robot.controllers.items():
        action_idx = robot.controller_action_idx.get(name)
        if action_idx is None or not isinstance(controller, JointController):
            continue

        command = q[controller.dof_idx]
        if isinstance(controller, HolonomicBaseJointController):
            cur_rz_joint_pos = cur_joint_pos[robot.base_idx][5]
            delta_q = wrap_angle(command[2] - cur_rz_joint_pos)
            canonical_pos = th.tensor([command[0], command[1], body_pos[2]], dtype=q.dtype, device=q.device)
            local_pos = T.relative_pose_transform(
                canonical_pos,
                th.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype, device=q.device),
                body_pos,
                body_quat,
            )[0]
            command = th.stack([local_pos[0], local_pos[1], delta_q])
        elif controller.use_delta_commands:
            if controller.motor_type != "position":
                raise ValueError(
                    f"Cannot convert absolute joint target to delta action for controller [{name}] "
                    f"with motor_type={controller.motor_type!r}"
                )
            command = command - cur_joint_pos[controller.dof_idx]

        action[action_idx] = controller._reverse_preprocess_command(command)

    return action


def _temporarily_clamp_curobo_joint_limits(cmg, emb_sel, joint_limit_overrides):
    """Temporarily clamp CuRobo cspace joint limits, returning a restore callback."""
    if not joint_limit_overrides:
        return None
    mg = getattr(cmg, "mg", {}).get(emb_sel)
    if mg is None:
        return None

    def _joint_limit_objects():
        holders = []
        seen_holders = set()

        def _add_holder(holder):
            if holder is None or id(holder) in seen_holders:
                return
            seen_holders.add(id(holder))
            holders.append(holder)

        def _add_rollout(rollout):
            if rollout is None:
                return
            _add_holder(getattr(getattr(rollout, "kinematics", None), "kinematics_config", None))
            dynamics_model = getattr(rollout, "dynamics_model", None)
            _add_holder(dynamics_model)
            _add_holder(getattr(getattr(dynamics_model, "robot_model", None), "kinematics_config", None))
            for attr in ("bound_cost", "bound_constraint"):
                _add_holder(getattr(rollout, attr, None))
            for cfg_attr in ("cost_cfg", "constraint_cfg"):
                bound_cfg = getattr(getattr(rollout, cfg_attr, None), "bound_cfg", None)
                _add_holder(bound_cfg)

        _add_holder(getattr(getattr(mg, "kinematics", None), "kinematics_config", None))
        for attr in ("rollout_fn",):
            _add_rollout(getattr(mg, attr, None))

        for solver_attr in (
            "ik_solver",
            "trajopt_solver",
            "finetune_trajopt_solver",
            "js_trajopt_solver",
            "finetune_js_trajopt_solver",
        ):
            solver = getattr(mg, solver_attr, None)
            if solver is None:
                continue
            _add_rollout(getattr(solver, "rollout_fn", None))
            _add_rollout(getattr(solver, "interpolate_rollout", None))
            inner_solver = getattr(solver, "solver", None)
            _add_rollout(getattr(inner_solver, "rollout_fn", None))
            _add_rollout(getattr(inner_solver, "safety_rollout", None))
            get_all_rollouts = getattr(solver, "get_all_rollout_instances", None)
            if get_all_rollouts is not None:
                try:
                    for rollout in get_all_rollouts():
                        _add_rollout(rollout)
                except Exception:
                    pass

        joint_limits_list = []
        seen_limits = set()
        for holder in holders:
            joint_limits = getattr(holder, "joint_limits", None)
            if (
                joint_limits is not None
                and id(joint_limits) not in seen_limits
                and hasattr(joint_limits, "joint_names")
                and hasattr(joint_limits, "position")
            ):
                seen_limits.add(id(joint_limits))
                joint_limits_list.append(joint_limits)
        return joint_limits_list

    original_limits = []
    for joint_limits in _joint_limit_objects():
        for joint_name, (lower, upper) in joint_limit_overrides.items():
            if joint_name not in joint_limits.joint_names:
                continue
            joint_idx = joint_limits.joint_names.index(joint_name)
            low_tensor = joint_limits.position[0][joint_idx].detach().clone()
            high_tensor = joint_limits.position[1][joint_idx].detach().clone()
            original_limits.append((joint_limits, joint_idx, low_tensor, high_tensor))
            joint_limits.position[0][joint_idx] = th.as_tensor(
                lower, dtype=joint_limits.position[0].dtype, device=joint_limits.position[0].device
            )
            joint_limits.position[1][joint_idx] = th.as_tensor(
                upper, dtype=joint_limits.position[1].dtype, device=joint_limits.position[1].device
            )

    if not original_limits:
        return None

    def _restore():
        for joint_limits, joint_idx, low_tensor, high_tensor in original_limits:
            joint_limits.position[0][joint_idx] = low_tensor
            joint_limits.position[1][joint_idx] = high_tensor

    _restore.num_joint_limit_sets = len({id(item[0]) for item in original_limits})
    return _restore


def _postprocess_action_compatible(env, action):
    """Postprocess primitive actions, skipping Tiago-only head tracking on R1/R1Pro."""
    try:
        return env.primitive._postprocess_action(action)
    except AssertionError as e:
        if "Tracking object with camera is currently only supported for Tiago" in str(e):
            return action
        raise


class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (np.array): 4x4 pose target for robot controller
            gripper_action (np.array): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1

    def merge_wp(self, other):
        """
        Merge another Waypoint object into this one.
        """
        self.pose = np.concatenate([self.pose, other.pose], axis=0)
        self.gripper_action = np.concatenate([self.gripper_action, other.gripper_action], axis=0)
        self.noise = min(self.noise, other.noise)
        # TODO: the noise here is set to 0, can be change to help reduce the sim to real gap due to the sensor observation noises
        self.noise = 0.0


class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be an list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses,
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        # handle scalar to numpy array conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)

    def merge(self, other):
        """
        Merge another WaypointSequence object into this one.
        """
        self.sequence += other.sequence

class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.

        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if (len(self.waypoint_sequences) == 0):
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([[gripper_action] for _ in range(num_steps)])
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            if last_waypoint.pose.shape[0] == 8:
                # here is when transforming the two arms altogher, should be corresponding to the bimanual-coordinated phase
                poses_left, num_steps_2_left = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[0:4, :],
                    pose_2=pose[0:4, :],
                    num_steps=num_steps,
                )
                poses_right, num_steps_2_right = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[4:, :],
                    pose_2=pose[4:, :],
                    num_steps=num_steps,
                )
                poses = np.concatenate([poses_left, poses_right], axis=1)
                assert num_steps_2_left == num_steps_2_right
                num_steps_2 = num_steps_2_left
            else:
                # suitable for single arm transformation
                poses, num_steps_2 = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose,
                    pose_2=pose,
                    num_steps=num_steps,
                )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = (need_interp or need_fixed)

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose, # 8x4
                    gripper_action=target_for_interpolation.gripper_action, #2,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                    bimanual=bimanual,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                    bimanual=bimanual,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def _pad_tensors(self, tensor1, tensor2):
        M, _ = tensor1.shape
        N, _ = tensor2.shape
        max_size = max(M, N)

        def pad_tensor(tensor, size):
            if tensor.shape[0] < size:
                last_row = tensor[-1].unsqueeze(0)  # Extract last row
                repeat_count = size - tensor.shape[0]
                padding = last_row.repeat(repeat_count, 1)  # Repeat last row
                tensor = th.cat([tensor, padding], dim=0)
            return tensor

        tensor1 = pad_tensor(tensor1, max_size)
        tensor2 = pad_tensor(tensor2, max_size)

        return tensor1, tensor2

    def _subsample_tensor(self, tensor, num_samples=8):
        N = tensor.shape[0]

        if N <= num_samples:
            return tensor  # If N is less than or equal to num_samples, return as is

        indices = th.linspace(0, N - 1, steps=num_samples).long()  # Evenly spaced indices
        return tensor[indices]

    def downsample_replay_traj(self, left_replay_waypoints, right_reaplay_waypoints, ds_ratio=1, asyn_ds_ratio=True):
        # downsample the replay waypoints to reduce the hesitation problem
        len_left_wp = len(left_replay_waypoints)
        len_right_wp = len(right_reaplay_waypoints)

        if ds_ratio == 1 or ds_ratio is None:
            print('the replay waypoints are not downsampled')
            return left_replay_waypoints, right_reaplay_waypoints

        # TODO: the grasping motion should not be downsampled??
        # detect whether the left and right gripper action are changing

        if asyn_ds_ratio:
            # asyn downsample the waypoints regarding the gripper actions
            print('asyn downsample the waypoints regarding the gripper actions')

            # check when grasping starts for both hands
            left_gripper_actions = np.array([waypoint.gripper_action[0] for waypoint in left_replay_waypoints])
            right_gripper_actions = np.array([waypoint.gripper_action[1] for waypoint in right_reaplay_waypoints])
            left_gripper_actions_diff = np.diff(left_gripper_actions)
            right_gripper_actions_diff = np.diff(right_gripper_actions)
            # check when the gripper actions are changing
            left_gripper_actions_diff_idx = np.where(left_gripper_actions_diff != 0)[0]
            right_gripper_actions_diff_idx = np.where(right_gripper_actions_diff != 0)[0]
            if left_gripper_actions_diff_idx.size == 0: left_gripper_actions_diff_idx = np.array([len_left_wp])
            if right_gripper_actions_diff_idx.size == 0: right_gripper_actions_diff_idx = np.array([len_left_wp])
            # get the min number of the changing points
            grasp_start_idx = np.min([left_gripper_actions_diff_idx[0], right_gripper_actions_diff_idx[0]])

            left_before_grasp_ds = left_replay_waypoints[:grasp_start_idx:ds_ratio]
            right_before_grasp_ds = right_reaplay_waypoints[:grasp_start_idx:ds_ratio]

            grasp_ds_ratio = 2
            left_after_grasp_ds = left_replay_waypoints[grasp_start_idx::grasp_ds_ratio]
            right_after_grasp_ds = right_reaplay_waypoints[grasp_start_idx::grasp_ds_ratio]

            # concatenate the downsampled waypoints
            left_reaplay_wp_ds = left_before_grasp_ds + left_after_grasp_ds
            right_reaplay_wp_ds = right_before_grasp_ds + right_after_grasp_ds

            assert len(left_reaplay_wp_ds) == len(right_reaplay_wp_ds)
            print('downsample wp from {} to {}'.format(len_left_wp, len(right_reaplay_wp_ds)))

        else:
            # uniformly downsample the waypoints
            print('uniformly downsample the waypoints')
            left_reaplay_wp_ds = left_replay_waypoints[::ds_ratio]
            right_reaplay_wp_ds = right_reaplay_waypoints[::ds_ratio]
            assert len(left_reaplay_wp_ds) == len(right_reaplay_wp_ds)
            print('downsample wp from {} to {}'.format(len_left_wp, len(right_reaplay_wp_ds)))

        return left_reaplay_wp_ds, right_reaplay_wp_ds

    def setup_phase_logs(self, phase_type, baseline=None):
        current_phase_logs = dict()
        current_phase_logs["phase_type"] = phase_type
        current_phase_logs["base_sampling_time"] = dict()
        current_phase_logs["base_mp_planning_time"] = dict()
        current_phase_logs["base_mp_execution_time"] = dict()

        current_phase_logs["arm_mp_planning_time"] = dict()
        current_phase_logs["arm_mp_execution_time"] = dict()
        current_phase_logs["arm_replay_execution_time"] = dict()
        if baseline == "mimicgen":
            current_phase_logs["arm_interp_execution_time"] = dict()

        current_phase_logs["full_retract_mp_planning_time"] = dict()
        current_phase_logs["full_retract_mp_execution_time"] = dict()
        current_phase_logs["torso_retract_mp_planning_time"] = dict()
        current_phase_logs["torso_retract_mp_execution_time"] = dict()
        current_phase_logs["full_retract_mp_err"] = dict()
        current_phase_logs["torso_retract_mp_err"] = dict()

        current_phase_logs["visibility_stats"] = dict()

        return current_phase_logs

    def obtain_attached_object(self, env, robot):
        grasp_action = {"left": 1.0, "right": 1.0}
        attached_obj = {}
        attached_obj_scale = {}
        for local_arm_side in ["left", "right"]:
            is_grasping = robot.is_grasping(arm=local_arm_side)
            # print("local_arm_side is_grasping: ", local_arm_side, is_grasping)
            if is_grasping == og.controllers.IsGraspingState.TRUE:
                grasp_action[local_arm_side] = -1.0
                # Find the object that the robot is grapsing in that arm
                task_relevant_objs = env._get_task_relevant_objs()
                for task_relevant_obj in task_relevant_objs:
                    # TODO: remove the stationay object hardcoding. Make it more general
                    if all(keyword not in task_relevant_obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                        is_grasping_candidate_obj = robot.is_grasping(arm=local_arm_side, candidate_obj=task_relevant_obj)
                        # print("local_arm_side is_grasping_candidate_obj: ", local_arm_side, is_grasping_candidate_obj, task_relevant_obj.root_link.name)
                        if is_grasping_candidate_obj == og.controllers.IsGraspingState.TRUE:
                            print(f"arm {local_arm_side} is_grasping {task_relevant_obj.root_link.name}")
                            attached_obj[f"{local_arm_side}_eef_link"] = task_relevant_obj.root_link
                            attached_obj_scale[f"{local_arm_side}_eef_link"] = 0.9
                            # robot can only be holding one object at a time
                            break
        retval = dict(
            grasp_action=grasp_action,
            attached_obj=attached_obj,
            attached_obj_scale=attached_obj_scale,
        )
        return retval

    def reset_visibility_counter(self, env):
        """
        Reset the visibility counter for each sensor.
        """
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                env.num_frames_with_obj_visible[shortened_sensor_name] = 0
        env.num_frames_with_obj_visible["any"] = 0

    def check_ref_obj_visibility(self, env, obs, obs_info, ref_obj):
        any_visible = False
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                seg_instance = obs[f"{env.robot_name}::{sensor_name}::seg_instance"]
                seg_instance_info = obs_info[f"{env.robot_name}"][sensor_name]["seg_instance"]
                obj_key = next((key for key, value in seg_instance_info.items() if value == ref_obj.name), None)
                if obj_key is None:
                    count = 0
                    # if shortened_sensor_name == "eyes":
                    #     print("not found")
                else:
                    count = (seg_instance == obj_key).sum().item()
                    # if shortened_sensor_name == "eyes":
                    #     print("found")
                if count > 0:
                    env.num_frames_with_obj_visible[shortened_sensor_name] += 1
                    any_visible = True

        if any_visible:
            env.num_frames_with_obj_visible["any"] += 1

    def execute_baseline(
        self,
        env,
        env_interface,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None,
        src_curr_phase_actions=None,
        src_curr_phase_base_pose=None,
        baseline=None,
    ):
        if object_ref["arm_right"] is None:
            ref_object = object_ref["arm_left"]
        elif object_ref["arm_left"] is None:
            ref_object = object_ref["arm_right"]
        else:
            ref_object = object_ref["arm_right"]

        ref_obj = None
        if ref_object is not None:
            if "torso" in ref_object:
                if isinstance(env.robot, Tiago):
                    torso_link_name = "torso_lift_link"
                elif isinstance(env.robot, R1):
                    torso_link_name = "torso_link4"
                else:
                    raise ValueError("Robot type not supported")
                ref_obj = env.env.robots[0].links[torso_link_name]
            else:
                ref_obj = env.env.scene.object_registry("name", ref_object)
            print("ref_obj: ", ref_obj.name)
        robot = env.env.robots[0]

        # TODO: implement early stopping on 1. collision 2. attached object misatch
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            init_state = og.sim.dump_state()
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            init_global_env_step = env.global_env_step
            nav_execution_start_time = time.time()
            init_arm_left_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
            init_arm_right_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            for temp_idx, src_action in enumerate(src_curr_phase_actions):

                # To skip initial stationary actions during human data collection
                if env.execution_phase_ind == 0 and temp_idx < env.start_nav_step:
                    continue
                action = env.primitive._empty_action()
                action[robot.base_action_idx] = th.tensor(src_action[robot.base_action_idx], dtype=th.float32)
                action[robot.arm_action_idx["left"]] = init_arm_left_pos
                action[robot.arm_action_idx["right"]] = init_arm_right_pos
                if attached_obj["left"] is not None:
                    action[robot.gripper_action_idx["left"]] = -1
                if attached_obj["right"] is not None:
                    action[robot.gripper_action_idx["right"]] = -1
                state = env.get_state()["states"]
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=action)
                env.step(action, video_writer)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                # Check reference object visibility
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)

            # apply a zero action
            action = env.primitive._empty_action()
            action[robot.base_action_idx] = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
            action[robot.arm_action_idx["left"]] = init_arm_left_pos
            action[robot.arm_action_idx["right"]] = init_arm_right_pos
            env.step(action, video_writer)

            nav_execution_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0] = round(nav_execution_finish_time - nav_execution_start_time, 2)
            print("nav execution time: ", phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for nav_repeat {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
            print(f"Visibility stats for nav_repeat any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"])

            MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
            left_mp_ranges = [0, 0]
            right_mp_ranges = [0, 0]
            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase).
            # In this case MP succeeded and phase was actually executed
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results

        else:
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type, baseline=baseline)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}

            debug_tracking_enabled = os.environ.get("MOMAGEN_DEBUG_TRACKING") == "1"
            tracking_target_ok_threshold = float(os.environ.get("MOMAGEN_TRACKING_TARGET_OK_THRESHOLD", "0.25") or 0.25)
            tracking_fail_threshold = float(os.environ.get("MOMAGEN_TRACKING_FAIL_THRESHOLD", "0.30") or 0.30)
            contact_nudge_target_dist = os.environ.get("MOMAGEN_CONTACT_NUDGE_TARGET_DIST")
            contact_nudge_target_dist = None if contact_nudge_target_dist in {None, ""} else float(contact_nudge_target_dist)
            contact_nudge_max_dist = float(os.environ.get("MOMAGEN_CONTACT_NUDGE_MAX_DIST", "0.35") or 0.35)
            contact_nudge_min_delta = float(os.environ.get("MOMAGEN_CONTACT_NUDGE_MIN_DELTA", "0.005") or 0.005)
            contact_nudge_max_delta = float(os.environ.get("MOMAGEN_CONTACT_NUDGE_MAX_DELTA", "0.05") or 0.05)
            contact_nudge_max_track_err = float(os.environ.get("MOMAGEN_CONTACT_NUDGE_MAX_TRACK_ERR", "0.05") or 0.05)
            contact_nudge_phase_filter = os.environ.get("MOMAGEN_CONTACT_NUDGE_PHASES", "")
            contact_nudge_stage_filter = os.environ.get("MOMAGEN_CONTACT_NUDGE_STAGES", "")
            contact_nudge_allowed_phases = {
                int(item.strip()) for item in contact_nudge_phase_filter.split(",") if item.strip()
            }
            contact_nudge_allowed_stages = {
                item.strip() for item in contact_nudge_stage_filter.split(",") if item.strip()
            }
            replay_source_base_action = os.environ.get("MOMAGEN_REPLAY_SOURCE_BASE_ACTION", "0") != "0"
            replay_source_base_action_threshold = float(os.environ.get("MOMAGEN_REPLAY_SOURCE_BASE_ACTION_THRESHOLD", "0.75") or 0.75)
            source_base_preapproach_steps = int(os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_STEPS", "0") or 0)

            def _as_np(x):
                if x is None:
                    return None
                if hasattr(x, "detach"):
                    x = x.detach()
                if hasattr(x, "cpu"):
                    x = x.cpu()
                return np.asarray(x, dtype=float)

            def _debug_tracking(stage, target_pose=None, action=None, replay_step=None, extra=None):
                """Log target-vs-actual EEF tracking at phase boundaries and replay checkpoints."""
                if not debug_tracking_enabled:
                    return None

                record = {
                    "phase": int(env.execution_phase_ind),
                    "stage": stage,
                    "replay_step": None if replay_step is None else int(replay_step),
                }
                target_pose_np = _as_np(target_pose) if target_pose is not None else None
                for arm_name, row_start in (("left", 0), ("right", 4)):
                    candidate = None
                    candidate_pos = None
                    eef_pos = None
                    target_pos = None
                    eef_to_candidate = None
                    target_to_candidate = None
                    actual_to_target = None
                    contact_data = []
                    grasp = None
                    candidate_grasp = None
                    action_norm = None

                    try:
                        ref_name = object_ref.get(f"arm_{arm_name}") if object_ref is not None else None
                        candidate = env.env.scene.object_registry("name", ref_name) if ref_name else None
                    except Exception:
                        candidate = None
                    try:
                        eef_pos = _as_np(robot.eef_links[arm_name].get_position_orientation()[0])
                    except Exception:
                        eef_pos = None
                    try:
                        candidate_pos = _as_np(candidate.get_position_orientation()[0]) if candidate is not None else None
                    except Exception:
                        candidate_pos = None
                    if target_pose_np is not None:
                        try:
                            target_pos = np.asarray(target_pose_np[row_start:row_start + 3, 3], dtype=float)
                        except Exception:
                            target_pos = None
                    try:
                        eef_to_candidate = (
                            float(np.linalg.norm(eef_pos - candidate_pos))
                            if eef_pos is not None and candidate_pos is not None
                            else None
                        )
                    except Exception:
                        eef_to_candidate = None
                    try:
                        target_to_candidate = (
                            float(np.linalg.norm(target_pos - candidate_pos))
                            if target_pos is not None and candidate_pos is not None
                            else None
                        )
                    except Exception:
                        target_to_candidate = None
                    try:
                        actual_to_target = (
                            float(np.linalg.norm(eef_pos - target_pos))
                            if eef_pos is not None and target_pos is not None
                            else None
                        )
                    except Exception:
                        actual_to_target = None
                    try:
                        contact_data, _ = robot._find_gripper_contacts(arm=arm_name)
                        contact_data = [str(item) for item in contact_data]
                    except Exception as e:
                        contact_data = [f"ERR:{e}"]
                    try:
                        grasp = str(robot.is_grasping(arm=arm_name))
                    except Exception as e:
                        grasp = f"ERR:{e}"
                    try:
                        candidate_grasp = str(robot.is_grasping(arm=arm_name, candidate_obj=candidate)) if candidate is not None else None
                    except Exception as e:
                        candidate_grasp = f"ERR:{e}"
                    if action is not None:
                        try:
                            idx = robot.controller_action_idx.get(f"arm_{arm_name}", robot.arm_action_idx[arm_name])
                            action_norm = float(np.linalg.norm(_as_np(action[idx])))
                        except Exception:
                            action_norm = None

                    prefix = arm_name
                    record[f"{prefix}_eef_to_candidate_dist"] = eef_to_candidate
                    record[f"{prefix}_target_to_candidate_dist"] = target_to_candidate
                    record[f"{prefix}_actual_to_target_dist"] = actual_to_target
                    record[f"{prefix}_contacts"] = contact_data
                    record[f"{prefix}_grasp"] = grasp
                    record[f"{prefix}_candidate_grasp"] = candidate_grasp
                    record[f"{prefix}_action_norm"] = action_norm
                    record[f"{prefix}_target_pos"] = target_pos.tolist() if target_pos is not None else None
                    record[f"{prefix}_eef_pos"] = eef_pos.tolist() if eef_pos is not None else None
                    record[f"{prefix}_candidate_pos"] = candidate_pos.tolist() if candidate_pos is not None else None

                left_target_ok = (
                    record.get("left_target_to_candidate_dist") is not None
                    and record["left_target_to_candidate_dist"] <= tracking_target_ok_threshold
                )
                left_tracking_bad = (
                    record.get("left_actual_to_target_dist") is not None
                    and record["left_actual_to_target_dist"] > tracking_fail_threshold
                )
                failure_stage = None
                if left_target_ok and left_tracking_bad:
                    failure_stage = "eef_tracking_failed_after_target_transform_ok"
                elif record.get("left_contacts") == [] and "TRUE" not in str(record.get("left_candidate_grasp")):
                    failure_stage = "grasp_precondition_failed_no_contact"
                record["failure_stage"] = failure_stage
                if extra:
                    record.update(extra)

                phase_logs.setdefault(env.execution_phase_ind, {}).setdefault("tracking_debug", []).append(record)
                print(
                    "[MOMAGEN_TRACKING] "
                    f"phase={record['phase']} stage={stage} replay_step={record['replay_step']} "
                    f"left_dist={record.get('left_eef_to_candidate_dist')} "
                    f"left_target_dist={record.get('left_target_to_candidate_dist')} "
                    f"left_track_err={record.get('left_actual_to_target_dist')} "
                    f"left_action_norm={record.get('left_action_norm')} "
                    f"left_contacts={record.get('left_contacts')[:3] if record.get('left_contacts') is not None else None} "
                    f"left_grasp={record.get('left_grasp')} left_candidate_grasp={record.get('left_candidate_grasp')} "
                    f"failure_stage={failure_stage}",
                    flush=True,
                )
                return record

            def _is_grasp_true(value):
                return "TRUE" in str(value).upper()

            def _apply_contact_nudge(pose, left_gripper_action=None, right_gripper_action=None, stage=None, replay_step=None):
                """Optionally pull a closing-gripper EEF target closer to its object center.

                This is a gated recovery path for mimicgen object-frame targets that are
                tracked accurately but remain just outside contact range in the generated
                scene. It is disabled unless MOMAGEN_CONTACT_NUDGE_TARGET_DIST is set.
                """
                if contact_nudge_target_dist is None:
                    return pose
                if contact_nudge_allowed_phases and env.execution_phase_ind not in contact_nudge_allowed_phases:
                    return pose
                if contact_nudge_allowed_stages and stage not in contact_nudge_allowed_stages:
                    return pose

                def _record_contact_nudge(arm_name, old_dist, delta, skipped=False, reason=None):
                    record = {
                        "phase": int(env.execution_phase_ind),
                        "stage": stage,
                        "replay_step": None if replay_step is None else int(replay_step),
                        "arm": arm_name,
                        "old_dist": None if old_dist is None else float(old_dist),
                        "target_dist": float(contact_nudge_target_dist),
                        "delta": None if delta is None else float(delta),
                        "skipped": bool(skipped),
                        "reason": reason,
                    }
                    phase_logs.setdefault(env.execution_phase_ind, {}).setdefault("contact_nudge", []).append(record)
                    if debug_tracking_enabled:
                        if skipped:
                            print(
                                "[MOMAGEN_CONTACT_NUDGE] "
                                f"phase={env.execution_phase_ind} stage={stage} replay_step={replay_step} "
                                f"arm={arm_name} skipped=True reason={reason} old_dist={old_dist} "
                                f"target_dist={contact_nudge_target_dist} delta={delta}",
                                flush=True,
                            )
                        else:
                            print(
                                "[MOMAGEN_CONTACT_NUDGE] "
                                f"phase={env.execution_phase_ind} stage={stage} replay_step={replay_step} "
                                f"arm={arm_name} old_dist={old_dist:.6f} target_dist={contact_nudge_target_dist:.6f} "
                                f"delta={delta:.6f}",
                                flush=True,
                            )

                pose_np = np.array(pose, copy=True)
                for arm_name, row_start, gripper_action in (
                    ("left", 0, left_gripper_action),
                    ("right", 4, right_gripper_action),
                ):
                    try:
                        should_close = gripper_action is not None and float(np.asarray(gripper_action).item()) < 0
                    except Exception:
                        should_close = False
                    if not should_close:
                        continue

                    try:
                        ref_name = object_ref.get(f"arm_{arm_name}") if object_ref is not None else None
                        candidate = env.env.scene.object_registry("name", ref_name) if ref_name else None
                        candidate_pos = _as_np(candidate.get_position_orientation()[0]) if candidate is not None else None
                    except Exception:
                        candidate_pos = None
                    if candidate_pos is None:
                        continue

                    target_pos = np.asarray(pose_np[row_start:row_start + 3, 3], dtype=float)
                    try:
                        eef_pos = _as_np(robot.eef_links[arm_name].get_position_orientation()[0])
                        track_err = float(np.linalg.norm(eef_pos - target_pos))
                    except Exception:
                        track_err = None
                    if track_err is None or track_err > contact_nudge_max_track_err:
                        _record_contact_nudge(arm_name, None, None, skipped=True, reason="track_err")
                        continue
                    vec = candidate_pos - target_pos
                    dist = float(np.linalg.norm(vec))
                    if dist <= 1e-8 or dist > contact_nudge_max_dist:
                        _record_contact_nudge(arm_name, dist, None, skipped=True, reason="distance_out_of_range")
                        continue
                    delta = dist - contact_nudge_target_dist
                    if delta <= contact_nudge_min_delta:
                        _record_contact_nudge(arm_name, dist, delta, skipped=True, reason="delta_too_small")
                        continue
                    if delta > contact_nudge_max_delta:
                        _record_contact_nudge(arm_name, dist, delta, skipped=True, reason="delta_too_large")
                        continue

                    target_pos = target_pos + (delta / dist) * vec
                    pose_np[row_start:row_start + 3, 3] = target_pos
                    _record_contact_nudge(arm_name, dist, delta)

                return pose_np

            def _target_requires_base_motion(pose):
                """Return whether the active target is too far for arm-only mimicgen replay."""
                # Source base replay is only intended for carry phases (e.g. D1
                # picking_up_trash phase_2), where an already-attached object must
                # be transported toward a far target.  Do not enable it for initial
                # grasp phases; copied source base commands can move the robot away
                # before contact is established.
                try:
                    pose_np = np.asarray(pose, dtype=float)
                except Exception:
                    return False

                for arm_name, row_start in (("left", 0), ("right", 4)):
                    if object_ref.get(f"arm_{arm_name}") is None:
                        continue
                    if attached_obj is None or attached_obj.get(arm_name) is None:
                        continue
                    try:
                        eef_pos = _as_np(robot.eef_links[arm_name].get_position_orientation()[0])
                        target_pos = np.asarray(pose_np[row_start:row_start + 3, 3], dtype=float)
                        if float(np.linalg.norm(eef_pos - target_pos)) > replay_source_base_action_threshold:
                            return True
                    except Exception:
                        continue
                return False

            def _copy_source_base_action(action, src_action_ind, reason=None):
                """Optionally copy the source-demo base command into an arm IK action.

                Mimicgen target-pose replay normally emits base no-op commands. For
                D1 picking_up_trash phase_2, the source segment contains base motion
                carrying the trash can toward the soda can; without this, the arm-only
                controller chases a target several meters away and saturates.
                """
                if not replay_source_base_action or src_curr_phase_actions is None:
                    return action
                try:
                    src_actions_np = np.asarray(src_curr_phase_actions)
                    if src_actions_np.size == 0:
                        return action
                    src_action_ind = int(np.clip(src_action_ind, 0, src_actions_np.shape[0] - 1))
                    base_idx = robot.controller_action_idx.get("base")
                    if base_idx is None:
                        return action
                    action[base_idx] = src_actions_np[src_action_ind][base_idx]
                    if debug_tracking_enabled and not getattr(self, "_momagen_logged_source_base_action", False):
                        print(
                            "[MOMAGEN_SOURCE_BASE_ACTION] "
                            f"phase={env.execution_phase_ind} reason={reason} src_action_ind={src_action_ind} "
                            f"base_action={np.asarray(src_actions_np[src_action_ind][base_idx]).tolist()}",
                            flush=True,
                        )
                        self._momagen_logged_source_base_action = True
                except Exception as e:
                    if debug_tracking_enabled:
                        print(f"[MOMAGEN_SOURCE_BASE_ACTION] skipped reason={reason} error={e}", flush=True)
                return action

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            wholebody_cover_replay_enabled = bool(
                int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_COVER_REPLAY", "0") or 0)
            )
            wholebody_cover_replay_phase_in_range = bool(
                int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP", "0") or 0)
                and int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MIN_PHASE", "0") or 0)
                <= int(env.execution_phase_ind)
                <= int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MAX_PHASE", "999999") or 999999)
            )
            if wholebody_cover_replay_enabled and wholebody_cover_replay_phase_in_range:
                original_mp_end_steps = [int(step) for step in cur_subtask_end_step_MP]
                cur_subtask_end_step_MP = [len(seq), len(seq)]
                phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp_cover_replay", []).append(
                    {
                        "enabled": True,
                        "applied": True,
                        "phase": int(env.execution_phase_ind),
                        "original_mp_end_steps": original_mp_end_steps,
                        "new_mp_end_steps": [int(step) for step in cur_subtask_end_step_MP],
                        "reason": "wholebody_plan_through_contact_replay",
                    }
                )

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")


            # If baseline is mimicgen, perform interpolation + replay
            if baseline == "mimicgen":
                # ========================================= ARM INTERPOLATION START =============================================
                step_size = 0.005
                current_left_eef_pose = robot.get_eef_pose("left")
                if object_ref["arm_left"] is None:
                    poses_left = th.tensor(T.pose2mat(current_left_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_left, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_left_eef_pose),
                        pose_2=th.tensor(left_waypoint.pose[:4], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_left = th.tensor(poses_left, dtype=th.float32)

                current_right_eef_pose = robot.get_eef_pose("right")
                if object_ref["arm_right"] is None:
                    poses_right = th.tensor(T.pose2mat(current_right_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_right, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_right_eef_pose),
                        pose_2=th.tensor(right_waypoint.pose[4:], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_right = th.tensor(poses_right, dtype=th.float32)

                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*current_left_eef_pose)
                    env.eef_current_marker_right.set_position_orientation(*current_right_eef_pose)
                    interp_target_left = T.mat2pose(poses_left[-1])
                    interp_target_right = T.mat2pose(poses_right[-1])
                    env.eef_goal_marker_left.set_position_orientation(*interp_target_left)
                    env.eef_goal_marker_right.set_position_orientation(*interp_target_right)


                print("len(poses_left): ", len(poses_left))
                print("len(poses_right): ", len(poses_right))

                # Perform padding
                if len(poses_left) < len(poses_right):
                    repeat_times = len(poses_right) - len(poses_left)
                    poses_left = th.cat((poses_left, poses_left[-1].repeat(repeat_times, 1, 1)))
                elif len(poses_right) < len(poses_left):
                    repeat_times = len(poses_left) - len(poses_right)
                    poses_right = th.cat((poses_right, poses_right[-1].repeat(repeat_times, 1, 1)))

                if len(poses_left) != len(poses_right):
                    assert len(poses_left) == len(poses_right)
                poses = np.concatenate([poses_left, poses_right], axis=1)
                base_motion_required = _target_requires_base_motion(poses[-1])
                self._momagen_logged_source_base_action = False
                if debug_tracking_enabled and replay_source_base_action:
                    print(
                        "[MOMAGEN_SOURCE_BASE_ACTION] "
                        f"phase={env.execution_phase_ind} enabled=True base_motion_required={base_motion_required} "
                        f"threshold={replay_source_base_action_threshold}",
                        flush=True,
                    )

                init_global_env_step = env.global_env_step
                arm_interp_start_time = time.time()
                active_mp_steps = [
                    int(step)
                    for step, arm_key in zip(cur_subtask_end_step_MP, ("arm_left", "arm_right"))
                    if step is not None and object_ref.get(arm_key) is not None
                ]
                interp_src_stop = max([0] + active_mp_steps)
                for interp_step, pose in enumerate(poses):
                    pose = _apply_contact_nudge(
                        pose,
                        left_gripper_action=left_waypoint.gripper_action[0],
                        right_gripper_action=right_waypoint.gripper_action[1],
                        stage="interp",
                    )
                    interp_action = env_interface.target_pose_to_action(target_pose=pose)

                    interp_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                    interp_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]
                    if base_motion_required and interp_src_stop > 0 and len(poses) > 1:
                        src_action_ind = round(interp_step * (interp_src_stop - 1) / (len(poses) - 1))
                        interp_action = _copy_source_base_action(interp_action, src_action_ind, reason="interp_far_target")

                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=interp_action)
                    env.step(interp_action, video_writer)
                    left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                    right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(interp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    if ref_obj is not None:
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                arm_interp_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0] = round(arm_interp_finish_time - arm_interp_start_time, 2)
                print("Time taken for arm interpolation: ", phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0])
                _debug_tracking("interp_end", target_pose=poses[-1], action=interp_action)

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_interp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_interp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"])

                # Setting the interpolation ranges
                MP_end_step_local_list = [local_env_step, local_env_step]
                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]
                # =============================================== ARM INTERPOLATION END ==================================================

            # If baseline is skillgen, perform mp + replay
            elif baseline == "skillgen":
                # =============================================== Arm MP Planning =============================================

                # If at least one hand has motion planner waypoints, plan the motion
                if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                    target_pos = {
                        robot.eef_link_names["left"]: left_waypoint_pos,
                        robot.eef_link_names["right"]: right_waypoint_pos,
                    }
                    target_quat = {
                        robot.eef_link_names["left"]: left_waypoint_ori,
                        robot.eef_link_names["right"]: right_waypoint_ori,
                    }
                    emb_sel = getattr(
                        CuRoboEmbodimentSelection,
                        "ARM_NO_TORSO",
                        CuRoboEmbodimentSelection.ARM,
                    )

                    # Use OG to know attached objects
                    retval = self.obtain_attached_object(env, robot)
                    attached_obj = retval["attached_obj"]
                    attached_obj_scale = retval["attached_obj_scale"]

                    # If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                    if object_ref["arm_right"] is None:
                        del target_pos["right_eef_link"]
                        del target_quat["right_eef_link"]
                    elif object_ref["arm_left"] is None:
                        del target_pos["left_eef_link"]
                        del target_quat["left_eef_link"]

                    print("ARM MP START")
                    eyes_target_pos, eyes_target_quat = None, None

                    if enable_marker_vis:
                        env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                        env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                        env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                        env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)

                    # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                    num_tries = 3
                    arm_mp_trial = 0
                    new_target_pos = copy.deepcopy(target_pos)
                    while True:

                        # Base condition
                        if arm_mp_trial > 0:
                            status_value = _mp_status_value(mp_results[0])

                            # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                            if env.retry_nav_on_arm_mp_failure:
                                base_condition = arm_mp_trial == num_tries
                            else:
                                base_condition = arm_mp_trial == num_tries or ("IK Fail" in status_value)

                            if base_condition:
                                print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                                if "TrajOpt Fail" in status_value:
                                    env.err = "ArmMPTrajOptFailed"
                                elif "IK Fail" in status_value:
                                    env.err = "ArmMPIKFailed"
                                else:
                                    env.err = "ArmMPOtherFailed"
                                env.valid_env = False
                                env.execution_phase_ind += 1
                                return None

                        # Aggregate target_pos and target_quat to match batch_size
                        new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                        new_target_quat = {
                            k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                        }

                        arm_mp_planning_start_time = time.time()
                        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                        mp_results, traj_paths = _compute_trajectories_with_paths(env.cmg,
                            target_pos=new_target_pos,
                            target_quat=new_target_quat,
                            is_local=False,
                            max_attempts=50,
                            timeout=60.0,
                            ik_fail_return=10,
                            enable_finetune_trajopt=True,
                            finetune_attempts=1,
                            return_full_result=True,
                            success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                            attached_obj=attached_obj,
                            attached_obj_scale=attached_obj_scale,
                            attached_obj_options=_attached_payload_options(attached_obj),
                            emb_sel=emb_sel,
                        )
                        arm_mp_planning_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                        successes = mp_results[0].success
                        print("Arm MP successes: ", successes)
                        success_idx = th.where(successes)[0].cpu()

                        if len(success_idx) == 0:
                            print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                            arm_mp_trial += 1
                            # modify target_pos a bit
                            for k in target_pos.keys():
                                new_target_pos[k] = target_pos[k] + th.rand(3) * 0.01 - 0.005
                            continue
                        else:
                            traj_path = traj_paths[success_idx[0]]
                            break

                    print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                    # ========================================================= End of Arm MP Planning ==========================================================

                    # ========================================================== Arm MP Execution ==========================================================
                    arm_mp_execution_start_time = time.time()

                    # Convert planned joint trajectory to actions
                    # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01)
                    q_traj = q_traj.cpu()
                    mp_actions = []
                    for j_pos in q_traj:

                        # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                        if object_ref["arm_left"] is None:
                            j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        elif object_ref["arm_right"] is None:
                            j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                        action = _joint_trajectory_point_to_action(robot, j_pos).cpu().numpy()

                        # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        if left_gripper_action is not None:
                            action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                        if right_gripper_action is not None:
                            action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]

                        mp_actions.append(action)

                    left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                    right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                    # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                    if len(left_mp_waypoints) == 0:
                        # We need to pad the left hand waypoints to match the length of the MP trajectory
                        if len(left_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                                left_replay_waypoints.append(last_waypoint)

                        left_eef_poses = []
                        # We convert the target pose of the left hand to replay_action
                        # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                            left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_left"]
                            action[action_idx] = replay_action[action_idx]
                            action[env_interface.gripper_action_dim[0]] = left_replay_waypoints[i].gripper_action[0]

                        # We remove the waypoints that have been replayed for the left arm
                        left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                    # Same logic as above but for the right hand
                    elif len(right_mp_waypoints) == 0:
                        if len(right_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                                right_replay_waypoints.append(last_waypoint)
                        right_eef_poses = []
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                            right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_right"]
                            action[action_idx] = replay_action[action_idx]
                            action[env_interface.gripper_action_dim[1]] = right_replay_waypoints[i].gripper_action[1]

                        right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                    assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                    init_global_env_step = env.global_env_step
                    num_repeat = 1
                    for i, mp_action in enumerate(mp_actions):
                        for _ in range(num_repeat):
                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            # TODO: Check if we can use primtiive stack execute action here. This will allow for checking convergence errors etc.
                            env.step(mp_action, video_writer)
                            if enable_marker_vis:
                                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                                env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                                env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            if ref_obj is not None:
                                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]

                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]


                MP_end_step_local = copy.deepcopy(local_env_step)
                # left MP points
                if len(left_mp_waypoints) == 0:
                    left_MP_end_step_local = 0
                else:
                    left_MP_end_step_local = MP_end_step_local
                if len(right_mp_waypoints) == 0:
                    right_MP_end_step_local = 0
                else:
                    right_MP_end_step_local = MP_end_step_local

                MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

                arm_mp_execution_finish_time = time.time()
                # Since there is only 1 trial for arm MP execution, we set the 0th index
                phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
                print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

                # ============================================== End of Arm MP ==========================================================

            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)

            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            left_replay_waypoints, right_replay_waypoints = self.downsample_replay_traj(
                left_replay_waypoints,
                right_replay_waypoints,
                ds_ratio=ds_ratio,
            )
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()
            prev_left_gripper_action = None
            prev_right_gripper_action = None

            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos

            # Optional mimicgen grasp dwell: hold the first replay target with the
            # closing gripper command before executing a source segment where the
            # object may already be attached / moving in the source demo. This is
            # especially important when the MP/replay boundary lands exactly on an
            # open->close transition: without a dwell, the target trajectory can
            # immediately follow the source "carry" motion while the object in the
            # generated scene is still ungrasped and stationary.
            grasp_dwell_steps = int(os.environ.get("MOMAGEN_GRASP_DWELL_STEPS", "0") or 0)
            grasp_dwell_stop_on_grasp = os.environ.get("MOMAGEN_GRASP_DWELL_STOP_ON_GRASP", "1") != "0"
            grasp_dwell_fail_fast = os.environ.get("MOMAGEN_GRASP_DWELL_FAIL_FAST", "1") != "0"
            if grasp_dwell_steps > 0 and (len(left_replay_waypoints) > 0 or len(right_replay_waypoints) > 0):
                first_left_waypoint = left_replay_waypoints[0] if len(left_replay_waypoints) > 0 else last_waypoint
                first_right_waypoint = right_replay_waypoints[0] if len(right_replay_waypoints) > 0 else last_waypoint
                left_should_dwell = object_ref.get("arm_left") is not None and float(np.asarray(first_left_waypoint.gripper_action[0]).item()) < 0
                right_should_dwell = object_ref.get("arm_right") is not None and float(np.asarray(first_right_waypoint.gripper_action[1]).item()) < 0
                if left_should_dwell or right_should_dwell:
                    print(f"[MOMAGEN_GRASP_DWELL] Holding first replay pose for up to {grasp_dwell_steps} step(s)")
                    dwell_success = {"left": False, "right": False}
                    last_dwell_pose = None
                    last_dwell_action = None
                    for dwell_step in range(grasp_dwell_steps):
                        pose = np.zeros((8, 4))
                        pose[:4, :] = first_left_waypoint.pose[:4, :]
                        pose[4:, :] = first_right_waypoint.pose[4:, :]
                        if object_ref["arm_right"] is None:
                            pose[4:, :] = current_right_ee_pose
                        elif object_ref["arm_left"] is None:
                            pose[:4, :] = current_left_ee_pose
                        pose = _apply_contact_nudge(
                            pose,
                            left_gripper_action=first_left_waypoint.gripper_action[0],
                            right_gripper_action=first_right_waypoint.gripper_action[1],
                            stage="dwell",
                            replay_step=dwell_step,
                        )
                        dwell_action = env_interface.target_pose_to_action(target_pose=pose)
                        dwell_action[env_interface.gripper_action_dim[0]] = first_left_waypoint.gripper_action[0]
                        dwell_action[env_interface.gripper_action_dim[1]] = first_right_waypoint.gripper_action[1]
                        last_dwell_pose = pose
                        last_dwell_action = dwell_action

                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=dwell_action)
                        env.step(dwell_action, video_writer)
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(dwell_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        if ref_obj is not None:
                            self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                        for arm_name, should_dwell in (("left", left_should_dwell), ("right", right_should_dwell)):
                            if not should_dwell:
                                continue
                            try:
                                candidate = env.env.scene.object_registry("name", object_ref[f"arm_{arm_name}"])
                                dwell_success[arm_name] = _is_grasp_true(robot.is_grasping(arm=arm_name, candidate_obj=candidate))
                            except Exception:
                                dwell_success[arm_name] = False
                        phase_logs.setdefault(env.execution_phase_ind, {}).setdefault("grasp_dwell", []).append({
                            "step": int(dwell_step),
                            "left_should_dwell": bool(left_should_dwell),
                            "right_should_dwell": bool(right_should_dwell),
                            "left_success": bool(dwell_success["left"]),
                            "right_success": bool(dwell_success["right"]),
                        })
                        print(
                            "[MOMAGEN_GRASP_DWELL] "
                            f"step={dwell_step + 1}/{grasp_dwell_steps} "
                            f"left_success={dwell_success['left']} right_success={dwell_success['right']}",
                            flush=True,
                        )
                        if grasp_dwell_stop_on_grasp:
                            left_done = (not left_should_dwell) or dwell_success["left"]
                            right_done = (not right_should_dwell) or dwell_success["right"]
                            if left_done and right_done:
                                break
                    if last_dwell_pose is not None:
                        _debug_tracking(
                            "dwell_end",
                            target_pose=last_dwell_pose,
                            action=last_dwell_action,
                            extra={
                                "left_dwell_success": bool(dwell_success["left"]),
                                "right_dwell_success": bool(dwell_success["right"]),
                            },
                        )
                    left_dwell_failed = left_should_dwell and not dwell_success["left"]
                    right_dwell_failed = right_should_dwell and not dwell_success["right"]
                    if grasp_dwell_fail_fast and (left_dwell_failed or right_dwell_failed):
                        phase_logs.setdefault(env.execution_phase_ind, {}).setdefault("grasp_dwell_fail_fast", []).append({
                            "left_should_dwell": bool(left_should_dwell),
                            "right_should_dwell": bool(right_should_dwell),
                            "left_success": bool(dwell_success["left"]),
                            "right_success": bool(dwell_success["right"]),
                        })
                        print(
                            "[MOMAGEN_GRASP_DWELL] "
                            "fail_fast=True aborting replay because grasp did not establish during dwell "
                            f"left_success={dwell_success['left']} right_success={dwell_success['right']}",
                            flush=True,
                        )
                        return None

            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            init_global_env_step = env.global_env_step
            active_mp_steps = [
                int(step)
                for step, arm_key in zip(cur_subtask_end_step_MP, ("arm_left", "arm_right"))
                if step is not None and object_ref.get(arm_key) is not None
            ]
            replay_src_start = max([0] + active_mp_steps)
            for replay_step, (left_waypoint, right_waypoint) in enumerate(zip(left_replay_waypoints, right_replay_waypoints)):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                    pose[:4, :] = current_left_ee_pose
                pose = _apply_contact_nudge(
                    pose,
                    left_gripper_action=left_waypoint.gripper_action[0],
                    right_gripper_action=right_waypoint.gripper_action[1],
                    stage="replay",
                    replay_step=replay_step,
                )
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                replay_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                replay_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]
                if 'base_motion_required' in locals() and base_motion_required:
                    replay_action = _copy_source_base_action(
                        replay_action,
                        replay_src_start + replay_step,
                        reason="replay_far_target",
                    )

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                if replay_step < 5 or replay_step % 50 == 0 or replay_step >= len(left_replay_waypoints) - 5:
                    _debug_tracking(
                        "replay_checkpoint",
                        target_pose=pose,
                        action=replay_action,
                        replay_step=replay_step,
                    )
                if os.environ.get("MOMAGEN_DEBUG_GRIPPER") == "1":
                    left_gripper_action = float(np.asarray(left_waypoint.gripper_action[0]).item())
                    right_gripper_action = float(np.asarray(right_waypoint.gripper_action[1]).item())
                    left_changed = prev_left_gripper_action is None or not np.isclose(left_gripper_action, prev_left_gripper_action)
                    right_changed = prev_right_gripper_action is None or not np.isclose(right_gripper_action, prev_right_gripper_action)
                    should_log = left_changed or right_changed or replay_step < 5 or replay_step % 50 == 0 or replay_step >= len(left_replay_waypoints) - 5
                    if should_log:
                        try:
                            left_candidate = env.env.scene.object_registry("name", object_ref["arm_left"]) if object_ref.get("arm_left") else None
                        except Exception:
                            left_candidate = None
                        try:
                            right_candidate = env.env.scene.object_registry("name", object_ref["arm_right"]) if object_ref.get("arm_right") else None
                        except Exception:
                            right_candidate = None
                        try:
                            left_grasp = robot.is_grasping(arm="left")
                        except Exception as e:
                            left_grasp = f"ERR:{e}"
                        try:
                            right_grasp = robot.is_grasping(arm="right")
                        except Exception as e:
                            right_grasp = f"ERR:{e}"
                        try:
                            left_candidate_grasp = robot.is_grasping(arm="left", candidate_obj=left_candidate) if left_candidate is not None else None
                        except Exception as e:
                            left_candidate_grasp = f"ERR:{e}"
                        try:
                            right_candidate_grasp = robot.is_grasping(arm="right", candidate_obj=right_candidate) if right_candidate is not None else None
                        except Exception as e:
                            right_candidate_grasp = f"ERR:{e}"
                        try:
                            left_eef_pos = robot.eef_links["left"].get_position_orientation()[0]
                            left_eef_pos = np.asarray(left_eef_pos.cpu() if hasattr(left_eef_pos, "cpu") else left_eef_pos, dtype=float)
                        except Exception:
                            left_eef_pos = None
                        try:
                            left_candidate_pos = left_candidate.get_position_orientation()[0] if left_candidate is not None else None
                            left_candidate_pos = np.asarray(
                                left_candidate_pos.cpu() if hasattr(left_candidate_pos, "cpu") else left_candidate_pos,
                                dtype=float,
                            ) if left_candidate_pos is not None else None
                        except Exception:
                            left_candidate_pos = None
                        try:
                            left_eef_to_candidate_dist = (
                                float(np.linalg.norm(left_eef_pos - left_candidate_pos))
                                if left_eef_pos is not None and left_candidate_pos is not None
                                else None
                            )
                        except Exception:
                            left_eef_to_candidate_dist = None
                        try:
                            left_contact_data, left_contact_links = robot._find_gripper_contacts(arm="left")
                            left_contact_data = [str(item) for item in left_contact_data]
                            left_contact_links = {str(k): [str(vv) for vv in v] for k, v in left_contact_links.items()}
                        except Exception as e:
                            left_contact_data = [f"ERR:{e}"]
                            left_contact_links = {}
                        try:
                            left_target_pos = np.asarray(pose[:3, 3], dtype=float)
                            left_target_to_candidate_dist = (
                                float(np.linalg.norm(left_target_pos - left_candidate_pos))
                                if left_candidate_pos is not None
                                else None
                            )
                            left_actual_to_target_dist = (
                                float(np.linalg.norm(left_eef_pos - left_target_pos))
                                if left_eef_pos is not None
                                else None
                            )
                        except Exception:
                            left_target_pos = None
                            left_target_to_candidate_dist = None
                            left_actual_to_target_dist = None
                        gripper_debug = {
                            "phase": int(env.execution_phase_ind),
                            "replay_step": int(replay_step),
                            "num_replay_steps": int(len(left_replay_waypoints)),
                            "gripper_dim": env_interface.gripper_action_dim.tolist(),
                            "left_cmd": left_gripper_action,
                            "left_action": float(replay_action[env_interface.gripper_action_dim[0]]),
                            "left_grasp": str(left_grasp),
                            "left_candidate": getattr(left_candidate, "name", None),
                            "left_candidate_grasp": str(left_candidate_grasp),
                            "left_eef_pos": left_eef_pos.tolist() if left_eef_pos is not None else None,
                            "left_target_pos": left_target_pos.tolist() if left_target_pos is not None else None,
                            "left_candidate_pos": left_candidate_pos.tolist() if left_candidate_pos is not None else None,
                            "left_eef_to_candidate_dist": left_eef_to_candidate_dist,
                            "left_target_to_candidate_dist": left_target_to_candidate_dist,
                            "left_actual_to_target_dist": left_actual_to_target_dist,
                            "left_contact_data": left_contact_data,
                            "left_contact_links": left_contact_links,
                            "right_cmd": right_gripper_action,
                            "right_action": float(replay_action[env_interface.gripper_action_dim[1]]),
                            "right_grasp": str(right_grasp),
                            "right_candidate": getattr(right_candidate, "name", None),
                            "right_candidate_grasp": str(right_candidate_grasp),
                        }
                        phase_logs.setdefault(env.execution_phase_ind, {}).setdefault("gripper_debug", []).append(gripper_debug)
                        print(
                            "[MOMAGEN_DEBUG_GRIPPER] "
                            f"phase={gripper_debug['phase']} replay_step={gripper_debug['replay_step']}/{gripper_debug['num_replay_steps']} "
                            f"gripper_dim={gripper_debug['gripper_dim']} "
                            f"left_cmd={gripper_debug['left_cmd']:.3f} left_action={gripper_debug['left_action']:.3f} "
                            f"left_grasp={gripper_debug['left_grasp']} left_candidate={gripper_debug['left_candidate']} "
                            f"left_candidate_grasp={gripper_debug['left_candidate_grasp']} "
                            f"left_dist={gripper_debug['left_eef_to_candidate_dist']} "
                            f"left_target_dist={gripper_debug['left_target_to_candidate_dist']} "
                            f"left_track_err={gripper_debug['left_actual_to_target_dist']} "
                            f"left_contacts={gripper_debug['left_contact_data'][:3]} "
                            f"right_cmd={gripper_debug['right_cmd']:.3f} right_action={gripper_debug['right_action']:.3f} "
                            f"right_grasp={gripper_debug['right_grasp']} right_candidate={gripper_debug['right_candidate']} "
                            f"right_candidate_grasp={gripper_debug['right_candidate_grasp']}",
                            flush=True,
                        )
                    prev_left_gripper_action = left_gripper_action
                    prev_right_gripper_action = right_gripper_action
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3], dtype=th.float32)))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3], dtype=th.float32)))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            _log_manip_debug(
                stage="after_arm_replay",
                target_pos_by_link=target_pos if "target_pos" in locals() else None,
                attached_obj_by_link=attached_obj if isinstance(attached_obj, dict) else None,
            )

            local_env_step = _maybe_execute_toggle_marker_post_mp_press(
                left_gripper_action=left_gripper_action,
                right_gripper_action=right_gripper_action,
                video_writer=video_writer,
                states=states,
                actions=actions,
                observations=observations,
                observations_info=observations_info,
                datagen_infos=datagen_infos,
                success=success,
                local_env_step=local_env_step,
                execute_live_q_to_action=execute_live_q_to_action,
                wholebody_arm_mp_enabled=wholebody_arm_mp_enabled,
                emb_sel=emb_sel,
                press_timing_stage="after_arm_replay",
            )

            # =================================================== End of Arm Replay ==========================================================

            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results



    def execute(
        self,
        env,
        env_interface,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None,
        src_curr_phase_actions=None,
        src_curr_phase_base_pose=None,
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (MG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
            cur_subtask_end_step_MP: list of size 2, the end point of motion planner for two arms

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """

        # TODO: This is duplicate code (also there in data_generator.py). Refactor this
        if object_ref["arm_right"] is None:
            ref_object = object_ref["arm_left"]
        elif object_ref["arm_left"] is None:
            ref_object = object_ref["arm_right"]
        else:
            ref_object = object_ref["arm_right"]

        if "torso" in ref_object:
            if isinstance(env.robot, Tiago):
                torso_link_name = "torso_lift_link"
            elif isinstance(env.robot, R1):
                torso_link_name = "torso_link4"
            else:
                raise ValueError("Robot type not supported")
            ref_obj = env.env.robots[0].links[torso_link_name]
        else:
            ref_obj = env.env.scene.object_registry("name", ref_object)
        env.primitive._tracking_object = ref_obj
        print("Will track object for this sub-step: ", ref_obj.name)
        robot = env.env.robots[0]
        source_base_preapproach_steps = int(os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_STEPS", "0") or 0)

        def _arm_has_active_payload(arm_name):
            """Return whether an arm should be considered active for MP / replay target preservation.

            Some coordinated BEHAVIOR phases use one arm as the semantic ``object_ref`` arm and the
            other as the carrying / attached-object arm.  Treating the attached arm as inactive freezes
            its source replay pose, preventing the held object (and any task marker on it) from following
            the demonstrated contact trajectory.
            """
            if object_ref is not None and object_ref.get(f"arm_{arm_name}") is not None:
                return True
            attached = attached_obj if isinstance(attached_obj, dict) else {}
            candidate_keys = [f"arm_{arm_name}", arm_name]
            try:
                candidate_keys.append(robot.eef_link_names[arm_name])
            except Exception:
                pass
            return any(attached.get(key) is not None for key in candidate_keys)

        def _arm_has_attached_payload(arm_name):
            attached = attached_obj if isinstance(attached_obj, dict) else {}
            candidate_keys = [f"arm_{arm_name}", arm_name]
            try:
                candidate_keys.append(robot.eef_link_names[arm_name])
            except Exception:
                pass
            return any(attached.get(key) is not None for key in candidate_keys)

        def _source_base_pose_to_2d_pose(source_base_pose):
            if source_base_pose is None:
                return None
            source_base_pose = np.asarray(source_base_pose, dtype=float)
            if source_base_pose.shape == (4, 4):
                yaw = T.quat2euler(T.mat2quat(th.as_tensor(source_base_pose[:3, :3], dtype=th.float32)))[2]
                return th.tensor([source_base_pose[0, 3], source_base_pose[1, 3], float(yaw)], dtype=th.float32)
            if source_base_pose.shape[-1] >= 3:
                return th.as_tensor(source_base_pose[:3], dtype=th.float32)
            return None

        def _source_base_pose_for_nav_arm(selected_nav_arm):
            if src_curr_phase_base_pose is None or cur_subtask_end_step_MP is None:
                return None, None
            try:
                if selected_nav_arm == "left":
                    source_base_idx = int(cur_subtask_end_step_MP[0])
                elif selected_nav_arm == "right":
                    source_base_idx = int(cur_subtask_end_step_MP[1])
                else:
                    source_base_idx = int(max(cur_subtask_end_step_MP))
                source_base_idx = min(max(source_base_idx, 0), len(src_curr_phase_base_pose) - 1)
                pose_2d = _source_base_pose_to_2d_pose(src_curr_phase_base_pose[source_base_idx])
                if pose_2d is None or not bool(th.isfinite(pose_2d).all()):
                    return None, source_base_idx
                return pose_2d, source_base_idx
            except Exception:
                return None, None

        def _snap_pose_to_current_traversable_component(pose_2d):
            """Snap a source-demo base pose to the nearest eroded-traversable pixel on the current component."""
            if pose_2d is None or not bool(int(os.environ.get("MOMAGEN_NAV_SOURCE_BASE_SNAP_TO_TRAV", "1") or 1)):
                return pose_2d, None
            try:
                import cv2

                trav_map = getattr(robot.scene, "trav_map", None) or getattr(robot.scene, "_trav_map", None)
                if trav_map is None or getattr(trav_map, "floor_map", None) is None:
                    return pose_2d, {"reason": "trav_map_unavailable"}
                robot_base_pos = robot.get_position_orientation()[0]
                floor_heights = getattr(trav_map, "floor_heights", None) or [0.0]
                floor = int(np.argmin(np.abs(np.asarray([float(h) for h in floor_heights]) - float(_debug_to_np(robot_base_pos)[2]))))
                raw_map = th.clone(trav_map.floor_map[floor])
                eroded_map = trav_map._erode_trav_map(th.clone(raw_map), robot=robot)
                _, labels = cv2.connectedComponents(eroded_map.cpu().numpy(), connectivity=4)

                start_xy = trav_map.world_to_map(th.as_tensor(_debug_to_np(robot_base_pos)[:2], dtype=th.float32))
                start_row, start_col = int(start_xy[0].item()), int(start_xy[1].item())
                if not (0 <= start_row < labels.shape[0] and 0 <= start_col < labels.shape[1]):
                    return pose_2d, {"reason": "start_out_of_bounds"}
                start_component = int(labels[start_row, start_col])
                if start_component == 0:
                    return pose_2d, {"reason": "start_not_on_eroded_component"}

                target_xy = trav_map.world_to_map(th.as_tensor(_debug_to_np(pose_2d)[:2], dtype=th.float32))
                target_row, target_col = int(target_xy[0].item()), int(target_xy[1].item())
                in_bounds = 0 <= target_row < labels.shape[0] and 0 <= target_col < labels.shape[1]
                if in_bounds and int(labels[target_row, target_col]) == start_component:
                    return pose_2d, {"reason": "already_on_current_component", "component": start_component}

                component_pixels_np = np.argwhere(labels == start_component)
                if component_pixels_np.size == 0:
                    return pose_2d, {"reason": "component_empty", "component": start_component}
                component_pixels = th.as_tensor(component_pixels_np, dtype=th.float32)
                target_pixel = th.tensor([target_row, target_col], dtype=th.float32)
                dists = th.norm(component_pixels - target_pixel.reshape(1, 2), dim=1)
                nearest_idx = int(th.argmin(dists).item())
                nearest_dist = float(dists[nearest_idx].item())
                max_pixel_dist = float(os.environ.get("MOMAGEN_NAV_SOURCE_BASE_SNAP_MAX_PIXELS", "10") or 10)
                if nearest_dist > max_pixel_dist:
                    return pose_2d, {
                        "reason": "nearest_component_pixel_too_far",
                        "component": start_component,
                        "nearest_pixel_dist": nearest_dist,
                        "max_pixel_dist": max_pixel_dist,
                    }
                nearest_map = component_pixels[nearest_idx].int()
                nearest_world = trav_map.map_to_world(nearest_map)
                snapped = pose_2d.clone()
                snapped[:2] = th.as_tensor(nearest_world[:2], dtype=snapped.dtype)
                return snapped, {
                    "reason": "snapped_to_current_component",
                    "component": start_component,
                    "source_map": [target_row, target_col],
                    "snapped_map": [int(nearest_map[0].item()), int(nearest_map[1].item())],
                    "snapped_world": _debug_array_value(nearest_world),
                    "nearest_pixel_dist": nearest_dist,
                    "max_pixel_dist": max_pixel_dist,
                }
            except Exception as e:
                return pose_2d, {"reason": "snap_exception", "error": f"{type(e).__name__}: {e}"}

        def _debug_to_np(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            return np.asarray(x, dtype=float)

        def _debug_pose_record(pose_getter, label, nonfinite_fields):
            try:
                pos, quat = pose_getter()
                pos_np = _debug_to_np(pos)
                quat_np = _debug_to_np(quat)
                record = {
                    "pos": None if pos_np is None else pos_np.tolist(),
                    "quat": None if quat_np is None else quat_np.tolist(),
                }
                for suffix, arr in (("pos", pos_np), ("quat", quat_np)):
                    if arr is not None and not bool(np.isfinite(arr).all()):
                        nonfinite_fields.append(f"{label}.{suffix}")
                return record
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}

        def _log_manip_debug(stage, target_pos_by_link=None, attached_obj_by_link=None, arm_mp_status=None):
            nonfinite_fields = []
            record = {
                "phase": int(env.execution_phase_ind),
                "stage": stage,
                "global_env_step": int(env.global_env_step),
                "robot_base_pose": _debug_pose_record(
                    robot.get_position_orientation, "robot_base_pose", nonfinite_fields
                ),
                "ref_obj_name": getattr(ref_obj, "name", None),
                "ref_obj_pose": _debug_pose_record(ref_obj.get_position_orientation, "ref_obj_pose", nonfinite_fields),
                "attached_obj_pose": {},
                "target_pos_by_link": {},
                "eef_pos_by_arm": {},
                "eef_target_dist": {},
                "target_ref_dist": {},
                "arm_mp_status": arm_mp_status,
            }

            def _safe_state_value(obj, state_cls, *args):
                try:
                    if obj is None or state_cls not in obj.states:
                        return None
                    value = obj.states[state_cls].get_value(*args)
                    if isinstance(value, (bool, np.bool_)):
                        return bool(value)
                    return str(value)
                except Exception as e:
                    return f"ERR:{type(e).__name__}: {e}"

            def _safe_adjacency(obj):
                try:
                    if obj is None or object_states.VerticalAdjacency not in obj.states:
                        return None
                    adjacency = obj.states[object_states.VerticalAdjacency].get_value()
                    return {
                        "positive_neighbors": [getattr(o, "name", str(o)) for o in adjacency.positive_neighbors],
                        "negative_neighbors": [getattr(o, "name", str(o)) for o in adjacency.negative_neighbors],
                    }
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            def _safe_toggle_debug(obj):
                """Return marker-level ToggledOn diagnostics without mutating simulator state."""
                try:
                    if obj is None or object_states.ToggledOn not in obj.states:
                        return None

                    toggle_state = obj.states[object_states.ToggledOn]
                    marker = getattr(toggle_state, "visual_marker", None)
                    marker_pos_np = None
                    marker_debug = {
                        "robot_can_toggle_steps": int(getattr(toggle_state, "robot_can_toggle_steps", -1)),
                        "obj_in_finger_contact_objs": None,
                        "visual_marker_pose": None,
                        "visual_marker_radius": None,
                        "eef_dist_to_marker": {},
                        "finger_min_dist_to_marker": {},
                    }

                    finger_contact_objs = getattr(object_states.ToggledOn, "_finger_contact_objs", None)
                    if finger_contact_objs is not None:
                        try:
                            marker_debug["obj_in_finger_contact_objs"] = obj in finger_contact_objs
                        except Exception as e:
                            marker_debug["obj_in_finger_contact_objs"] = f"ERR:{type(e).__name__}: {e}"

                    if marker is not None:
                        marker_debug["visual_marker_pose"] = _debug_pose_record(
                            marker.get_position_orientation,
                            f"toggle_debug.{getattr(obj, 'name', 'obj')}.visual_marker_pose",
                            nonfinite_fields,
                        )
                        marker_pos_np = _debug_to_np(marker.get_position_orientation()[0])
                        try:
                            extent_np = _debug_to_np(getattr(marker, "extent", None))
                            scale_np = _debug_to_np(getattr(marker, "scale", None))
                            if extent_np is not None and scale_np is not None:
                                marker_debug["visual_marker_radius"] = float(np.min(extent_np * scale_np))
                        except Exception as e:
                            marker_debug["visual_marker_radius"] = f"ERR:{type(e).__name__}: {e}"

                    for arm in ("left", "right"):
                        marker_debug["eef_dist_to_marker"][arm] = None
                        marker_debug["finger_min_dist_to_marker"][arm] = None
                        if marker_pos_np is None:
                            continue

                        try:
                            eef_pos_np = _debug_to_np(robot.eef_links[arm].get_position_orientation()[0])
                            marker_debug["eef_dist_to_marker"][arm] = float(np.linalg.norm(eef_pos_np - marker_pos_np))
                        except Exception as e:
                            marker_debug["eef_dist_to_marker"][arm] = f"ERR:{type(e).__name__}: {e}"

                        try:
                            finger_links = getattr(robot, "finger_links", {}).get(arm, [])
                            finger_records = []
                            for finger_link in finger_links:
                                finger_pos_np = _debug_to_np(finger_link.get_position_orientation()[0])
                                finger_records.append(
                                    {
                                        "link": getattr(finger_link, "name", str(finger_link)),
                                        "dist": float(np.linalg.norm(finger_pos_np - marker_pos_np)),
                                    }
                                )
                            if finger_records:
                                closest = min(finger_records, key=lambda item: item["dist"])
                                marker_debug["finger_min_dist_to_marker"][arm] = closest
                        except Exception as e:
                            marker_debug["finger_min_dist_to_marker"][arm] = f"ERR:{type(e).__name__}: {e}"

                    return marker_debug
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            def _maybe_apply_toggle_marker_target_correction(target_pos, target_quat):
                """Diagnostic press-target correction: aim the active finger, not the EEF origin, at the toggle marker."""
                if not bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_CORRECTION", "0") or 0)):
                    return
                min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MIN_PHASE", "0") or 0)
                max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MAX_PHASE", "999999") or 999999)
                if not (min_phase <= int(env.execution_phase_ind) <= max_phase):
                    return
                if ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                    return

                active_arms = []
                for arm_name in ("left", "right"):
                    link_name = robot.eef_link_names[arm_name]
                    arm_ref = object_ref.get(f"arm_{arm_name}")
                    ref_name = getattr(ref_obj, "name", None)
                    if (
                        (arm_ref is ref_obj or getattr(arm_ref, "name", None) == ref_name or str(arm_ref) == str(ref_name))
                        and link_name in target_pos
                    ):
                        active_arms.append(arm_name)
                if not active_arms:
                    return

                try:
                    toggle_state = ref_obj.states[object_states.ToggledOn]
                    marker = getattr(toggle_state, "visual_marker", None)
                    if marker is None:
                        return
                    marker_pos = th.as_tensor(marker.get_position_orientation()[0], dtype=th.float32)
                except Exception as e:
                    phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                        {"enabled": True, "applied": False, "reason": f"marker_error:{type(e).__name__}: {e}"}
                    )
                    return

                for arm_name in active_arms:
                    try:
                        eef_link_name = robot.eef_link_names[arm_name]
                        eef_pos = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype)
                        finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                        if not finger_links:
                            continue

                        finger_records = []
                        for finger_link in finger_links:
                            finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                            finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                        finger_link, finger_pos, finger_dist = min(finger_records, key=lambda item: item[2])

                        # Keep the current EEF-to-finger world offset as a first-order proxy. This is deliberately
                        # diagnostic-only; it checks whether reaching the physical toggle marker fixes the predicate.
                        offset_scale = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_OFFSET_SCALE", "1.0") or 1.0)
                        approach_z = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_APPROACH_Z", "0.0") or 0.0)
                        eef_to_finger_offset = eef_pos - finger_pos
                        corrected_pos = marker_pos + offset_scale * eef_to_finger_offset
                        corrected_pos = corrected_pos + th.tensor([0.0, 0.0, approach_z], dtype=corrected_pos.dtype)
                        corrected_pos = corrected_pos.to(dtype=target_pos[eef_link_name].dtype, device=target_pos[eef_link_name].device)

                        original_pos = target_pos[eef_link_name]
                        target_pos[eef_link_name] = corrected_pos
                        correction_record = {
                            "enabled": True,
                            "applied": True,
                            "phase": int(env.execution_phase_ind),
                            "arm": arm_name,
                            "eef_link": eef_link_name,
                            "finger_link": getattr(finger_link, "name", str(finger_link)),
                            "marker_pos": _debug_array_value(marker_pos),
                            "finger_pos": _debug_array_value(finger_pos),
                            "eef_pos": _debug_array_value(eef_pos),
                            "finger_marker_dist_before": finger_dist,
                            "original_target_pos": _debug_array_value(original_pos),
                            "corrected_target_pos": _debug_array_value(corrected_pos),
                            "offset_scale": offset_scale,
                            "approach_z": approach_z,
                        }
                        phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                            correction_record
                        )
                        print("[MOMAGEN_TOGGLE_MARKER_TARGET_CORRECTION] " + json.dumps(correction_record, default=str), flush=True)
                    except Exception as e:
                        phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                            {
                                "enabled": True,
                                "applied": False,
                                "phase": int(env.execution_phase_ind),
                                "arm": arm_name,
                                "reason": f"correction_error:{type(e).__name__}: {e}",
                            }
                        )

            try:
                cur_success_metrics = env.is_success()
                record["success_metrics"] = {
                    str(k): (bool(v) if isinstance(v, (bool, np.bool_)) else str(v))
                    for k, v in cur_success_metrics.items()
                }
            except Exception as e:
                record["success_metrics"] = {"error": f"{type(e).__name__}: {e}"}

            try:
                record["attached_object_names"] = self.obtain_attached_object(env, robot)
                # The raw attached_obj payload contains link objects; keep only serializable names.
                record["attached_object_names"]["attached_obj"] = {
                    str(k): getattr(v, "name", str(v))
                    for k, v in record["attached_object_names"].get("attached_obj", {}).items()
                }
            except Exception as e:
                record["attached_object_names"] = {"error": f"{type(e).__name__}: {e}"}

            try:
                task_relevant_objs = list(env._get_task_relevant_objs())
            except Exception:
                task_relevant_objs = []
            record["object_state_debug"] = {}
            for obj in task_relevant_objs:
                obj_name = getattr(obj, "name", None)
                if obj_name is None:
                    continue
                obj_record = {
                    "pose": _debug_pose_record(obj.get_position_orientation, f"object_state_debug.{obj_name}", nonfinite_fields),
                    "toggled_on": _safe_state_value(obj, object_states.ToggledOn),
                    "vertical_adjacency": _safe_adjacency(obj),
                    "toggle_debug": _safe_toggle_debug(obj),
                }
                if ref_obj is not None and obj is not ref_obj:
                    obj_record["touching_ref_obj"] = _safe_state_value(obj, object_states.Touching, ref_obj)
                    obj_record["ontop_ref_obj"] = _safe_state_value(obj, object_states.OnTop, ref_obj)
                record["object_state_debug"][obj_name] = obj_record

            for link_name, obj_link in (attached_obj_by_link or {}).items():
                if obj_link is None:
                    continue
                record["attached_obj_pose"][link_name] = {
                    "name": getattr(obj_link, "name", None),
                    "pose": _debug_pose_record(
                        obj_link.get_position_orientation,
                        f"attached_obj_pose.{link_name}",
                        nonfinite_fields,
                    ),
                }

            for arm in ("left", "right"):
                try:
                    eef_link_name = robot.eef_link_names[arm]
                    target = None if target_pos_by_link is None else target_pos_by_link.get(eef_link_name)
                    if target is None:
                        record["eef_target_dist"][arm] = None
                        record["target_ref_dist"][arm] = None
                        continue

                    eef_pos = _debug_to_np(robot.eef_links[arm].get_position_orientation()[0])
                    target_np = _debug_to_np(target)
                    ref_pos_np = _debug_to_np(ref_obj.get_position_orientation()[0])

                    if eef_pos is not None and not bool(np.isfinite(eef_pos).all()):
                        nonfinite_fields.append(f"{arm}.eef_pos")
                    if target_np is not None and not bool(np.isfinite(target_np).all()):
                        nonfinite_fields.append(f"{arm}.target_pos")
                    if ref_pos_np is not None and not bool(np.isfinite(ref_pos_np).all()):
                        nonfinite_fields.append(f"{arm}.ref_pos_for_target_dist")

                    record["eef_pos_by_arm"][arm] = None if eef_pos is None else eef_pos.tolist()
                    record["target_pos_by_link"][eef_link_name] = None if target_np is None else target_np.tolist()

                    record["eef_target_dist"][arm] = (
                        None
                        if eef_pos is None or target_np is None
                        else float(np.linalg.norm(eef_pos - target_np))
                    )
                    record["target_ref_dist"][arm] = (
                        None
                        if target_np is None or ref_pos_np is None
                        else float(np.linalg.norm(target_np - ref_pos_np))
                    )
                except Exception as e:
                    record["eef_target_dist"][arm] = f"ERR:{type(e).__name__}: {e}"
                    record["target_ref_dist"][arm] = f"ERR:{type(e).__name__}: {e}"

            record["has_nonfinite"] = bool(nonfinite_fields)
            record["nonfinite_fields"] = nonfinite_fields

            phase_logs[env.execution_phase_ind].setdefault("manip_debug", []).append(record)
            print("[MOMAGEN_MANIP_DEBUG] " + json.dumps(record, default=str), flush=True)
            return record

        def _maybe_apply_toggle_marker_target_correction(target_pos, target_quat):
            """Diagnostic press-target correction: aim the active finger, not the EEF origin, at the toggle marker."""
            if not bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_CORRECTION", "0") or 0)):
                return
            min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MIN_PHASE", "0") or 0)
            max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MAX_PHASE", "999999") or 999999)
            if not (min_phase <= int(env.execution_phase_ind) <= max_phase):
                return
            if ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                return

            active_arms = []
            for arm_name in ("left", "right"):
                link_name = robot.eef_link_names[arm_name]
                arm_ref = object_ref.get(f"arm_{arm_name}")
                ref_name = getattr(ref_obj, "name", None)
                if (
                    (arm_ref is ref_obj or getattr(arm_ref, "name", None) == ref_name or str(arm_ref) == str(ref_name))
                    and link_name in target_pos
                ):
                    active_arms.append(arm_name)
            active_arms_override_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_TARGET_ACTIVE_ARMS_OVERRIDE", ""
            ).strip()
            if active_arms_override_raw:
                requested_active_arms = [
                    value.strip().lower()
                    for value in active_arms_override_raw.split(",")
                    if value.strip()
                ]
                invalid_active_arms = [
                    arm_name for arm_name in requested_active_arms if arm_name not in ("left", "right")
                ]
                if invalid_active_arms:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_TARGET_ACTIVE_ARMS_OVERRIDE only supports "
                        f"'left' and/or 'right', got {invalid_active_arms}"
                    )
                active_arms = [
                    arm_name
                    for arm_name in requested_active_arms
                    if robot.eef_link_names[arm_name] in target_pos
                ]
            if not active_arms:
                return

            try:
                toggle_state = ref_obj.states[object_states.ToggledOn]
                marker = getattr(toggle_state, "visual_marker", None)
                if marker is None:
                    return
                marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
                marker_pos = th.as_tensor(marker_pos_raw, dtype=th.float32)
                marker_quat = th.as_tensor(marker_quat_raw, dtype=th.float32)
                marker_rot = th.as_tensor(T.quat2mat(marker_quat), dtype=marker_pos.dtype)
            except Exception as e:
                phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                    {"enabled": True, "applied": False, "reason": f"marker_error:{type(e).__name__}: {e}"}
                )
                return

            for arm_name in active_arms:
                try:
                    eef_link_name = robot.eef_link_names[arm_name]
                    payload_hold_records = []
                    if bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_HOLD_PAYLOAD_ARM", "0") or 0)):
                        # For toggle tasks, the payload arm often only needs to keep the object stable while the
                        # other hand presses the marker.  Source-demo EEF targets can otherwise transport the held
                        # object away from the live marker during the same whole-body solve, so optionally replace
                        # carrier-arm targets with the current live EEF pose and let the holonomic base / press arm
                        # solve around that stable object pose.
                        for carrier_arm in ("right", "left"):
                            if carrier_arm == arm_name or not _arm_has_active_payload(carrier_arm):
                                continue
                            carrier_link_name = robot.eef_link_names.get(carrier_arm)
                            if carrier_link_name not in target_pos or carrier_link_name not in target_quat:
                                continue
                            carrier_pos, carrier_quat = robot.get_eef_pose(carrier_arm)
                            original_carrier_pos = target_pos[carrier_link_name]
                            original_carrier_quat = target_quat[carrier_link_name]
                            target_pos[carrier_link_name] = th.as_tensor(
                                carrier_pos,
                                dtype=original_carrier_pos.dtype,
                                device=original_carrier_pos.device,
                            )
                            target_quat[carrier_link_name] = th.as_tensor(
                                carrier_quat,
                                dtype=original_carrier_quat.dtype,
                                device=original_carrier_quat.device,
                            )
                            payload_hold_records.append(
                                {
                                    "enabled": True,
                                    "applied": True,
                                    "arm": carrier_arm,
                                    "eef_link": carrier_link_name,
                                    "original_target_pos": _debug_array_value(original_carrier_pos),
                                    "held_target_pos": _debug_array_value(target_pos[carrier_link_name]),
                                    "original_target_quat": _debug_array_value(original_carrier_quat),
                                    "held_target_quat": _debug_array_value(target_quat[carrier_link_name]),
                                }
                            )

                    marker_target_pos = marker_pos
                    marker_prediction_record = None
                    if bool(
                        int(
                            os.environ.get(
                                "MOMAGEN_TOGGLE_MARKER_TARGET_USE_FUTURE_ATTACHED_MARKER", "0"
                            )
                            or 0
                        )
                    ):
                        # In coordinated carry+press phases, the press arm should aim at the marker pose
                        # after the carrier arm transports the attached object to its MP target.  The live
                        # marker before ARM MP is still at the pre-transport pose, so compute a first-order
                        # prediction by preserving the current carrier-EEF -> marker transform and applying
                        # it to the carrier arm's MP target pose.
                        for carrier_arm in ("right", "left"):
                            if carrier_arm == arm_name or not _arm_has_active_payload(carrier_arm):
                                continue
                            carrier_link_name = robot.eef_link_names.get(carrier_arm)
                            if carrier_link_name not in target_pos or carrier_link_name not in target_quat:
                                continue
                            carrier_pos, carrier_quat = robot.get_eef_pose(carrier_arm)
                            carrier_pos = th.as_tensor(carrier_pos, dtype=marker_pos.dtype, device=marker_pos.device)
                            carrier_quat = th.as_tensor(carrier_quat, dtype=marker_pos.dtype, device=marker_pos.device)
                            carrier_target_pos = th.as_tensor(
                                target_pos[carrier_link_name], dtype=marker_pos.dtype, device=marker_pos.device
                            )
                            carrier_target_quat = th.as_tensor(
                                target_quat[carrier_link_name], dtype=marker_pos.dtype, device=marker_pos.device
                            )
                            carrier_rot = T.quat2mat(carrier_quat)
                            carrier_target_rot = T.quat2mat(carrier_target_quat)
                            marker_local = carrier_rot.T @ (marker_pos - carrier_pos)
                            predicted_marker_pos = carrier_target_pos + carrier_target_rot @ marker_local
                            marker_target_pos = predicted_marker_pos
                            future_live_delta = float(th.linalg.norm(predicted_marker_pos - marker_pos))
                            carrier_target_delta = float(th.linalg.norm(carrier_target_pos - carrier_pos))
                            future_expected_delta = carrier_target_delta + float(th.linalg.norm(marker_local))
                            future_max_delta = float(
                                os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_FUTURE_MAX_DELTA", "0") or 0
                            )
                            future_on_fail = os.environ.get(
                                "MOMAGEN_TOGGLE_MARKER_TARGET_FUTURE_ON_FAIL", "fallback_live"
                            )
                            if future_on_fail not in ("fallback_live", "keep"):
                                future_on_fail = "fallback_live"
                            future_validation_enabled = future_max_delta > 0.0
                            future_validation_failed = bool(
                                future_validation_enabled
                                and (
                                    future_live_delta > future_max_delta
                                    or future_expected_delta > future_max_delta
                                    or not bool(th.isfinite(marker_target_pos).all())
                                )
                            )
                            future_used_predicted = not (
                                future_validation_failed and future_on_fail == "fallback_live"
                            )
                            if not future_used_predicted:
                                marker_target_pos = marker_pos
                            marker_prediction_record = {
                                "mode": "future_attached_marker",
                                "carrier_arm": carrier_arm,
                                "carrier_link": carrier_link_name,
                                "live_marker_pos": _debug_array_value(marker_pos),
                                "carrier_pos": _debug_array_value(carrier_pos),
                                "carrier_target_pos": _debug_array_value(carrier_target_pos),
                                "marker_local_in_carrier": _debug_array_value(marker_local),
                                "predicted_marker_pos": _debug_array_value(predicted_marker_pos),
                                "used_marker_target_pos": _debug_array_value(marker_target_pos),
                                "validation_enabled": future_validation_enabled,
                                "validation_failed": future_validation_failed,
                                "validation_on_fail": future_on_fail,
                                "validation_used_predicted": future_used_predicted,
                                "validation_max_delta": future_max_delta,
                                "predicted_live_marker_delta": future_live_delta,
                                "carrier_target_delta": carrier_target_delta,
                                "marker_local_norm": float(th.linalg.norm(marker_local)),
                                "expected_delta_bound": future_expected_delta,
                            }
                            break
                    eef_pos = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype)
                    finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                    if not finger_links:
                        continue

                    finger_records = []
                    for finger_link in finger_links:
                        finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                        finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                    force_finger_link = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_TARGET_FORCE_FINGER_LINK", ""
                    ).strip()
                    selected_finger_record = None
                    if force_finger_link:
                        for finger_record in finger_records:
                            finger_name = getattr(finger_record[0], "name", str(finger_record[0]))
                            if force_finger_link == finger_name or force_finger_link in finger_name:
                                selected_finger_record = finger_record
                                break
                    finger_link, finger_pos, finger_dist = selected_finger_record or min(
                        finger_records, key=lambda item: item[2]
                    )

                    finger_body_name = getattr(finger_link, "body_name", None)
                    if finger_body_name is None:
                        finger_body_name = str(getattr(finger_link, "name", finger_link)).split(":")[-1]

                    offset_scale = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_OFFSET_SCALE", "1.0") or 1.0)
                    approach_z = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_APPROACH_Z", "0.0") or 0.0)
                    marker_local_offset_raw = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_TARGET_MARKER_LOCAL_OFFSET", ""
                    ).strip()
                    marker_local_offset = None
                    if marker_local_offset_raw:
                        marker_local_offset_values = [
                            float(value.strip())
                            for value in marker_local_offset_raw.split(",")
                            if value.strip()
                        ]
                        if len(marker_local_offset_values) != 3:
                            raise ValueError(
                                "MOMAGEN_TOGGLE_MARKER_TARGET_MARKER_LOCAL_OFFSET must contain "
                                "3 comma-separated floats"
                            )
                        marker_local_offset = th.tensor(marker_local_offset_values, dtype=marker_pos.dtype)
                        marker_target_pos = marker_target_pos + marker_rot @ marker_local_offset
                    eef_to_finger_offset = eef_pos - finger_pos
                    corrected_pos = marker_target_pos + offset_scale * eef_to_finger_offset
                    corrected_pos = corrected_pos + th.tensor([0.0, 0.0, approach_z], dtype=corrected_pos.dtype)
                    max_correction_norm = float(
                        os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MAX_CORRECTION_NORM", "0.0") or 0.0
                    )
                    correction_delta = corrected_pos - eef_pos
                    correction_delta_norm = float(th.linalg.norm(correction_delta))
                    correction_delta_clamped = False
                    if max_correction_norm > 0.0 and correction_delta_norm > max_correction_norm:
                        corrected_pos = eef_pos + correction_delta * (max_correction_norm / correction_delta_norm)
                        correction_delta_clamped = True
                    corrected_pos = corrected_pos.to(dtype=target_pos[eef_link_name].dtype, device=target_pos[eef_link_name].device)

                    finger_link_goal_enabled = bool(
                        int(os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_FINGER_LINK_GOAL", "0") or 0)
                    )
                    finger_link_goal_record = None
                    if finger_link_goal_enabled:
                        additional_links = set(getattr(env.cmg, "additional_links", {}).get(emb_sel, []))
                        if finger_body_name in additional_links:
                            finger_target_pos = marker_target_pos + th.tensor(
                                [0.0, 0.0, approach_z], dtype=marker_target_pos.dtype, device=marker_target_pos.device
                            )
                            finger_target_pos = finger_target_pos.to(
                                dtype=target_pos[eef_link_name].dtype, device=target_pos[eef_link_name].device
                            )
                            finger_quat = th.as_tensor(
                                finger_link.get_position_orientation()[1],
                                dtype=target_quat[eef_link_name].dtype,
                                device=target_quat[eef_link_name].device,
                            )
                            target_pos[finger_body_name] = finger_target_pos
                            target_quat[finger_body_name] = finger_quat
                            finger_link_goal_record = {
                                "enabled": True,
                                "applied": True,
                                "link": finger_body_name,
                                "target_pos": _debug_array_value(finger_target_pos),
                                "target_quat": _debug_array_value(finger_quat),
                            }
                        else:
                            finger_link_goal_record = {
                                "enabled": True,
                                "applied": False,
                                "link": finger_body_name,
                                "reason": "finger_link_not_in_curobo_additional_links",
                                "available_additional_links": sorted(additional_links),
                            }

                    original_pos = target_pos[eef_link_name]
                    original_quat = target_quat.get(eef_link_name)
                    orientation_mode = str(
                        os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_ORIENTATION_MODE", "") or ""
                    ).strip().lower()
                    orientation_record = {
                        "mode": orientation_mode,
                        "applied": False,
                        "original_target_quat": _debug_array_value(original_quat),
                    }
                    if orientation_mode:
                        if orientation_mode == "current_eef":
                            current_eef_quat = th.as_tensor(
                                robot.eef_links[arm_name].get_position_orientation()[1],
                                dtype=target_quat[eef_link_name].dtype,
                                device=target_quat[eef_link_name].device,
                            )
                            target_quat[eef_link_name] = current_eef_quat
                            orientation_record.update(
                                {
                                    "applied": True,
                                    "target_quat": _debug_array_value(target_quat[eef_link_name]),
                                }
                            )
                        elif orientation_mode == "current_finger":
                            current_finger_quat = th.as_tensor(
                                finger_link.get_position_orientation()[1],
                                dtype=target_quat[eef_link_name].dtype,
                                device=target_quat[eef_link_name].device,
                            )
                            target_quat[eef_link_name] = current_finger_quat
                            orientation_record.update(
                                {
                                    "applied": True,
                                    "target_quat": _debug_array_value(target_quat[eef_link_name]),
                                }
                            )
                        else:
                            raise ValueError(
                                "MOMAGEN_TOGGLE_MARKER_TARGET_ORIENTATION_MODE only supports "
                                "current_eef/current_finger"
                            )
                    target_pos[eef_link_name] = corrected_pos
                    correction_record = {
                        "enabled": True,
                        "applied": True,
                        "phase": int(env.execution_phase_ind),
                        "arm": arm_name,
                        "active_arms_override": active_arms_override_raw or None,
                        "eef_link": eef_link_name,
                        "finger_link": getattr(finger_link, "name", str(finger_link)),
                        "force_finger_link": force_finger_link or None,
                        "finger_body_name": finger_body_name,
                        "marker_pos": _debug_array_value(marker_pos),
                        "marker_quat": _debug_array_value(marker_quat),
                        "marker_frame_axes_world": {
                            "x": _debug_array_value(marker_rot[:, 0]),
                            "y": _debug_array_value(marker_rot[:, 1]),
                            "z": _debug_array_value(marker_rot[:, 2]),
                        },
                        "marker_target_pos": _debug_array_value(marker_target_pos),
                        "marker_local_offset": _debug_array_value(marker_local_offset),
                        "marker_prediction": marker_prediction_record,
                        "finger_pos": _debug_array_value(finger_pos),
                        "eef_pos": _debug_array_value(eef_pos),
                        "finger_marker_dist_before": finger_dist,
                        "original_target_pos": _debug_array_value(original_pos),
                        "corrected_target_pos": _debug_array_value(corrected_pos),
                        "correction_delta_norm": correction_delta_norm,
                        "max_correction_norm": max_correction_norm,
                        "correction_delta_clamped": correction_delta_clamped,
                        "offset_scale": offset_scale,
                        "approach_z": approach_z,
                        "orientation": orientation_record,
                        "payload_hold": payload_hold_records,
                        "finger_link_goal": finger_link_goal_record,
                    }
                    phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                        correction_record
                    )
                    print("[MOMAGEN_TOGGLE_MARKER_TARGET_CORRECTION] " + json.dumps(correction_record, default=str), flush=True)
                except Exception as e:
                    phase_logs[env.execution_phase_ind].setdefault("toggle_marker_target_correction", []).append(
                        {
                            "enabled": True,
                            "applied": False,
                            "phase": int(env.execution_phase_ind),
                            "arm": arm_name,
                            "reason": f"correction_error:{type(e).__name__}: {e}",
                        }
                    )

        def _maybe_apply_toggle_marker_joint_staging_targets(target_pos, target_quat, emb_sel):
            """Jointly choose active-finger precontact and held-object staging targets for toggle tasks."""
            if not bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_TARGETS", "0") or 0)):
                return
            min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_MIN_PHASE", "0") or 0)
            max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_MAX_PHASE", "999999") or 999999)
            phase_in_range = min_phase <= int(env.execution_phase_ind) <= max_phase
            record = {
                "enabled": True,
                "phase": int(env.execution_phase_ind),
                "phase_type": phase_type,
                "phase_in_range": bool(phase_in_range),
                "emb_sel": str(emb_sel),
            }
            if not phase_in_range:
                record.update({"applied": False, "reason": "phase_out_of_range"})
            elif phase_type != "coordinated":
                record.update({"applied": False, "reason": "phase_type_not_coordinated"})
            elif ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                record.update({"applied": False, "reason": "ref_obj_not_toggleable"})
            elif not is_default_embodiment(emb_sel):
                record.update({"applied": False, "reason": "non_default_embodiment"})
            else:
                try:
                    active_arm = os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_ACTIVE_ARM", "left").strip().lower()
                    hold_arm = os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_HOLD_ARM", "right").strip().lower()
                    if active_arm not in robot.eef_link_names or hold_arm not in robot.eef_link_names:
                        raise ValueError(f"invalid active/hold arm: {active_arm}/{hold_arm}")
                    active_link_name = robot.eef_link_names[active_arm]
                    hold_link_name = robot.eef_link_names[hold_arm]
                    if active_link_name not in target_pos or hold_link_name not in target_pos:
                        raise ValueError("missing_active_or_hold_target")

                    toggle_state = ref_obj.states[object_states.ToggledOn]
                    marker = getattr(toggle_state, "visual_marker", None)
                    if marker is None:
                        raise ValueError("missing_visual_marker")
                    marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
                    marker_pos = th.as_tensor(marker_pos_raw, dtype=th.float32)
                    marker_quat = th.as_tensor(marker_quat_raw, dtype=th.float32)
                    marker_rot = T.quat2mat(marker_quat)

                    marker_local_offset_raw = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_MARKER_LOCAL_OFFSET",
                        os.environ.get(
                            "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MARKER_LOCAL_OFFSET",
                            os.environ.get(
                                "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_LOCAL_OFFSET",
                                os.environ.get(
                                    "MOMAGEN_TOGGLE_MARKER_TARGET_MARKER_LOCAL_OFFSET",
                                    "0.044,-0.035,0.013",
                                ),
                            ),
                        ),
                    )
                    marker_local_offset_values = [
                        float(value.strip())
                        for value in marker_local_offset_raw.split(",")
                        if value.strip()
                    ]
                    if len(marker_local_offset_values) != 3:
                        raise ValueError("marker_local_offset must contain 3 comma-separated floats")
                    marker_local_offset = th.tensor(marker_local_offset_values, dtype=marker_pos.dtype)

                    force_finger_link = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_FORCE_FINGER_LINK",
                        os.environ.get(
                            "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FORCE_FINGER_LINK",
                            os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_FORCE_FINGER_LINK", ""),
                        ),
                    ).strip()
                    finger_links = getattr(robot, "finger_links", {}).get(active_arm, [])
                    finger_records = []
                    for finger_link in finger_links:
                        finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                        finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                    if not finger_records:
                        raise ValueError("missing_active_finger_links")
                    selected_finger_record = None
                    if force_finger_link:
                        for finger_record in finger_records:
                            finger_name = getattr(finger_record[0], "name", str(finger_record[0]))
                            if force_finger_link == finger_name or force_finger_link in finger_name:
                                selected_finger_record = finger_record
                                break
                    finger_link, finger_pos, finger_dist = selected_finger_record or min(
                        finger_records,
                        key=lambda item: item[2],
                    )

                    active_eef_pos, active_eef_quat = robot.get_eef_pose(active_arm)
                    hold_pos, hold_quat = robot.get_eef_pose(hold_arm)
                    active_eef_pos = th.as_tensor(active_eef_pos, dtype=marker_pos.dtype)
                    active_eef_quat = th.as_tensor(active_eef_quat, dtype=marker_pos.dtype)
                    hold_pos = th.as_tensor(hold_pos, dtype=marker_pos.dtype)
                    hold_quat = th.as_tensor(hold_quat, dtype=marker_pos.dtype)
                    hold_rot = T.quat2mat(hold_quat)

                    active_target_pos = th.as_tensor(
                        target_pos[active_link_name],
                        dtype=marker_pos.dtype,
                        device=marker_pos.device,
                    )
                    eef_to_finger_offset = active_eef_pos - finger_pos
                    desired_finger_pos = active_target_pos - eef_to_finger_offset

                    hold_orientation_mode = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_HOLD_ORIENTATION_MODE",
                        "current_hold",
                    ).strip().lower()
                    if hold_orientation_mode == "current_hold":
                        desired_hold_quat = hold_quat
                    elif hold_orientation_mode == "target_hold":
                        desired_hold_quat = th.as_tensor(
                            target_quat[hold_link_name],
                            dtype=marker_pos.dtype,
                            device=marker_pos.device,
                        )
                    else:
                        raise ValueError(
                            "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_HOLD_ORIENTATION_MODE "
                            "only supports current_hold/target_hold"
                        )
                    desired_hold_rot = T.quat2mat(desired_hold_quat)

                    marker_local_in_hold = hold_rot.T @ (marker_pos - hold_pos)
                    marker_rot_in_hold = hold_rot.T @ marker_rot
                    desired_marker_rot = desired_hold_rot @ marker_rot_in_hold
                    desired_marker_pos = desired_finger_pos - desired_marker_rot @ marker_local_offset
                    desired_hold_pos = desired_marker_pos - desired_hold_rot @ marker_local_in_hold

                    max_hold_delta = float(
                        os.environ.get("MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_MAX_HOLD_DELTA", "0.0") or 0.0
                    )
                    hold_delta = desired_hold_pos - hold_pos
                    hold_delta_norm = float(th.linalg.norm(hold_delta))
                    hold_delta_clamped = False
                    if max_hold_delta > 0.0 and hold_delta_norm > max_hold_delta:
                        desired_hold_pos = hold_pos + hold_delta * (max_hold_delta / hold_delta_norm)
                        hold_delta_clamped = True

                    active_orientation_mode = os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_ACTIVE_ORIENTATION_MODE",
                        "keep",
                    ).strip().lower()
                    if active_orientation_mode == "current_eef":
                        target_quat[active_link_name] = active_eef_quat.to(
                            dtype=target_quat[active_link_name].dtype,
                            device=target_quat[active_link_name].device,
                        )
                    elif active_orientation_mode not in ("", "keep"):
                        raise ValueError(
                            "MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_ACTIVE_ORIENTATION_MODE "
                            "only supports keep/current_eef"
                        )

                    original_hold_pos = target_pos[hold_link_name]
                    original_hold_quat = target_quat[hold_link_name]
                    target_pos[hold_link_name] = desired_hold_pos.to(
                        dtype=original_hold_pos.dtype,
                        device=original_hold_pos.device,
                    )
                    target_quat[hold_link_name] = desired_hold_quat.to(
                        dtype=original_hold_quat.dtype,
                        device=original_hold_quat.device,
                    )

                    record.update(
                        {
                            "applied": True,
                            "active_arm": active_arm,
                            "hold_arm": hold_arm,
                            "active_link": active_link_name,
                            "hold_link": hold_link_name,
                            "finger_link": getattr(finger_link, "name", str(finger_link)),
                            "force_finger_link": force_finger_link or None,
                            "finger_marker_dist_before": finger_dist,
                            "marker_pos": _debug_array_value(marker_pos),
                            "marker_quat": _debug_array_value(marker_quat),
                            "marker_local_offset": _debug_array_value(marker_local_offset),
                            "active_eef_pos": _debug_array_value(active_eef_pos),
                            "active_target_pos": _debug_array_value(active_target_pos),
                            "finger_pos": _debug_array_value(finger_pos),
                            "eef_to_finger_offset": _debug_array_value(eef_to_finger_offset),
                            "desired_finger_pos": _debug_array_value(desired_finger_pos),
                            "hold_pos": _debug_array_value(hold_pos),
                            "hold_quat": _debug_array_value(hold_quat),
                            "marker_local_in_hold": _debug_array_value(marker_local_in_hold),
                            "marker_rot_in_hold": _debug_array_value(marker_rot_in_hold),
                            "desired_marker_pos": _debug_array_value(desired_marker_pos),
                            "desired_hold_pos": _debug_array_value(target_pos[hold_link_name]),
                            "desired_hold_quat": _debug_array_value(target_quat[hold_link_name]),
                            "original_hold_target_pos": _debug_array_value(original_hold_pos),
                            "original_hold_target_quat": _debug_array_value(original_hold_quat),
                            "hold_orientation_mode": hold_orientation_mode,
                            "active_orientation_mode": active_orientation_mode,
                            "hold_delta_norm": hold_delta_norm,
                            "max_hold_delta": max_hold_delta,
                            "hold_delta_clamped": hold_delta_clamped,
                        }
                    )
                except Exception as e:
                    record.update(
                        {
                            "applied": False,
                            "reason": f"joint_staging_error:{type(e).__name__}: {e}",
                        }
                    )
            phase_logs[env.execution_phase_ind].setdefault("toggle_marker_joint_staging_targets", []).append(record)
            print("[MOMAGEN_TOGGLE_MARKER_JOINT_STAGING_TARGETS] " + json.dumps(record, default=str), flush=True)

        def _maybe_apply_toggle_marker_replay_correction(pose, replay_step=None):
            """Diagnostic replay-time correction that keeps the active finger aimed at the live toggle marker.

            Unlike the ARM-MP target correction above, this runs inside the contact-rich replay loop. It is
            intentionally env-gated because it is task-specific and should only be used to test whether the
            remaining failure is replay/object-drift rather than planner feasibility.
            """
            if not bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_CORRECTION", "0") or 0)):
                return pose
            min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_MIN_PHASE", "0") or 0)
            max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_MAX_PHASE", "999999") or 999999)
            if not (min_phase <= int(env.execution_phase_ind) <= max_phase):
                return pose
            if ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                return pose

            pose_np = np.array(pose, copy=True)
            active_arms = []
            ref_name = getattr(ref_obj, "name", None)
            for arm_name in ("left", "right"):
                arm_ref = object_ref.get(f"arm_{arm_name}")
                if arm_ref is ref_obj or getattr(arm_ref, "name", None) == ref_name or str(arm_ref) == str(ref_name):
                    active_arms.append(arm_name)
            if not active_arms:
                return pose_np

            try:
                toggle_state = ref_obj.states[object_states.ToggledOn]
                marker = getattr(toggle_state, "visual_marker", None)
                if marker is None:
                    return pose_np
                marker_pos = th.as_tensor(marker.get_position_orientation()[0], dtype=th.float32)
            except Exception as e:
                phase_logs[env.execution_phase_ind].setdefault("toggle_marker_replay_correction", []).append(
                    {
                        "enabled": True,
                        "applied": False,
                        "phase": int(env.execution_phase_ind),
                        "replay_step": None if replay_step is None else int(replay_step),
                        "reason": f"marker_error:{type(e).__name__}: {e}",
                    }
                )
                return pose_np

            offset_scale = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_OFFSET_SCALE", "1.0") or 1.0)
            approach_z = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_APPROACH_Z", "0.0") or 0.0)
            log_interval = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_REPLAY_LOG_INTERVAL", "10") or 10)
            for arm_name in active_arms:
                try:
                    row_start = 0 if arm_name == "left" else 4
                    eef_pos = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype)
                    finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                    if not finger_links:
                        continue

                    finger_records = []
                    for finger_link in finger_links:
                        finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                        finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                    finger_link, finger_pos, finger_dist = min(finger_records, key=lambda item: item[2])

                    original_pos = np.array(pose_np[row_start:row_start + 3, 3], copy=True)
                    corrected_pos = marker_pos + offset_scale * (eef_pos - finger_pos)
                    corrected_pos = corrected_pos + th.tensor([0.0, 0.0, approach_z], dtype=corrected_pos.dtype)
                    pose_np[row_start:row_start + 3, 3] = _debug_to_np(corrected_pos)

                    should_log = replay_step is None or replay_step == 0 or (log_interval > 0 and replay_step % log_interval == 0)
                    if should_log:
                        record = {
                            "enabled": True,
                            "applied": True,
                            "phase": int(env.execution_phase_ind),
                            "replay_step": None if replay_step is None else int(replay_step),
                            "arm": arm_name,
                            "finger_link": getattr(finger_link, "name", str(finger_link)),
                            "marker_pos": _debug_array_value(marker_pos),
                            "finger_pos": _debug_array_value(finger_pos),
                            "eef_pos": _debug_array_value(eef_pos),
                            "finger_marker_dist_before": finger_dist,
                            "original_target_pos": _debug_array_value(original_pos),
                            "corrected_target_pos": _debug_array_value(corrected_pos),
                            "offset_scale": offset_scale,
                            "approach_z": approach_z,
                        }
                        phase_logs[env.execution_phase_ind].setdefault("toggle_marker_replay_correction", []).append(record)
                        print("[MOMAGEN_TOGGLE_MARKER_REPLAY_CORRECTION] " + json.dumps(record, default=str), flush=True)
                except Exception as e:
                    phase_logs[env.execution_phase_ind].setdefault("toggle_marker_replay_correction", []).append(
                        {
                            "enabled": True,
                            "applied": False,
                            "phase": int(env.execution_phase_ind),
                            "replay_step": None if replay_step is None else int(replay_step),
                            "arm": arm_name,
                            "reason": f"correction_error:{type(e).__name__}: {e}",
                        }
                    )
            return pose_np

        def _maybe_execute_toggle_marker_contact_prealign(
            left_gripper_action,
            right_gripper_action,
            video_writer,
            states,
            actions,
            observations,
            observations_info,
            datagen_infos,
            success,
            local_env_step,
            execute_live_q_to_action,
            wholebody_arm_mp_enabled,
            emb_sel,
            timing_stage="after_arm_replay",
            status_out=None,
        ):
            """Env-gated DEFAULT whole-body replan that moves the press finger into the live marker basin."""
            if not bool(int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN", "0") or 0)):
                return local_env_step
            log_key = "toggle_marker_contact_prealign"

            def _set_status(**kwargs):
                if status_out is not None:
                    status_out.update(kwargs)

            def _record(record):
                record.setdefault("enabled", True)
                record.setdefault("phase", int(env.execution_phase_ind))
                record.setdefault("timing_stage", timing_stage)
                phase_logs.setdefault(env.execution_phase_ind, {}).setdefault(log_key, []).append(record)
                print("[MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN] " + json.dumps(record, default=str), flush=True)

            if not (
                execute_live_q_to_action
                and wholebody_arm_mp_enabled
                and is_default_embodiment(emb_sel)
            ):
                _record(
                    {
                        "applied": False,
                        "reason": "requires_default_wholebody_live_q_to_action",
                        "execute_live_q_to_action": bool(execute_live_q_to_action),
                        "wholebody_arm_mp_enabled": bool(wholebody_arm_mp_enabled),
                        "emb_sel": str(emb_sel),
                    }
                )
                _set_status(attempted=False, passed=None, reason="requires_default_wholebody_live_q_to_action")
                return local_env_step
            min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MIN_PHASE", "0") or 0)
            max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_PHASE", "999999") or 999999)
            if not (min_phase <= int(env.execution_phase_ind) <= max_phase):
                _record(
                    {
                        "applied": False,
                        "reason": "phase_out_of_range",
                        "min_phase": min_phase,
                        "max_phase": max_phase,
                    }
                )
                _set_status(attempted=False, passed=None, reason="phase_out_of_range")
                return local_env_step
            if ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                _record({"applied": False, "reason": "missing_toggle_ref_obj"})
                _set_status(attempted=False, passed=False, reason="missing_toggle_ref_obj")
                return local_env_step

            desired_timing = (
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_TIMING", "after_arm_replay")
                or "after_arm_replay"
            ).strip().lower()
            if desired_timing != timing_stage:
                _set_status(attempted=False, passed=None, reason="timing_stage_mismatch")
                return local_env_step

            try:
                toggle_state = ref_obj.states[object_states.ToggledOn]
                marker = getattr(toggle_state, "visual_marker", None)
                if marker is None:
                    _record({"applied": False, "reason": "missing_visual_marker"})
                    _set_status(attempted=False, passed=False, reason="missing_visual_marker")
                    return local_env_step
                marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
                marker_pos = th.as_tensor(marker_pos_raw, dtype=th.float32)
                marker_quat = th.as_tensor(marker_quat_raw, dtype=th.float32)
                marker_rot = th.as_tensor(T.quat2mat(marker_quat), dtype=marker_pos.dtype)
            except Exception as e:
                _record({"applied": False, "reason": f"marker_error:{type(e).__name__}: {e}"})
                _set_status(attempted=False, passed=False, reason=f"marker_error:{type(e).__name__}: {e}")
                return local_env_step

            def _parse_vec3(env_name, default_value):
                raw = (os.environ.get(env_name, default_value) or default_value).strip()
                values = [float(value.strip()) for value in raw.split(",") if value.strip()]
                if len(values) != 3:
                    raise ValueError(f"{env_name} must contain 3 comma-separated floats")
                return th.tensor(values, dtype=marker_pos.dtype, device=marker_pos.device)

            marker_local_offset = _parse_vec3(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MARKER_LOCAL_OFFSET",
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_LOCAL_OFFSET",
                    os.environ.get("MOMAGEN_TOGGLE_MARKER_TARGET_MARKER_LOCAL_OFFSET", "0.044,-0.035,0.013"),
                ),
            )
            world_offset = _parse_vec3("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_WORLD_OFFSET", "0,0,0")
            approach_z = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_APPROACH_Z", "0.0") or 0.0)
            offset_scale = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_OFFSET_SCALE", "1.0") or 1.0)
            target_fraction = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_TARGET_FRACTION", "1.0") or 1.0
            )
            target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
            force_finger_link = os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FORCE_FINGER_LINK", "").strip()
            active_arms_raw = os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ACTIVE_ARMS", "left").strip()
            active_arms = [value.strip() for value in active_arms_raw.split(",") if value.strip()]
            invalid_arms = [arm_name for arm_name in active_arms if arm_name not in ("left", "right")]
            if invalid_arms:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ACTIVE_ARMS must be a comma-separated "
                    f"subset of left,right; got {invalid_arms}"
                )
            if not active_arms:
                _record({"applied": False, "reason": "no_active_arms"})
                _set_status(attempted=False, passed=False, reason="no_active_arms")
                return local_env_step

            def _select_finger(arm_name):
                finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                if not finger_links:
                    return None, None, []
                finger_records = []
                for finger_link in finger_links:
                    finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                    finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                if force_finger_link:
                    for finger_record in finger_records:
                        finger_name = getattr(finger_record[0], "name", str(finger_record[0]))
                        if force_finger_link == finger_name or force_finger_link in finger_name:
                            return finger_record[0], finger_record[1], finger_records
                selected = min(finger_records, key=lambda item: item[2])
                return selected[0], selected[1], finger_records

            def _arm_snapshot(arm_name, finger_link=None, snapshot_marker_pos=None, snapshot_marker_rot=None):
                marker_pos_for_snapshot = marker_pos if snapshot_marker_pos is None else snapshot_marker_pos
                marker_rot_for_snapshot = marker_rot if snapshot_marker_rot is None else snapshot_marker_rot
                eef_pos = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype)
                eef_quat = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[1], dtype=marker_pos.dtype)
                snapshot = {
                    "arm": arm_name,
                    "eef_pos": _debug_array_value(eef_pos),
                    "eef_quat": _debug_array_value(eef_quat),
                }
                if finger_link is not None:
                    finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                    marker_local = marker_rot_for_snapshot.T @ (finger_pos - marker_pos_for_snapshot)
                    source_residual = marker_local - marker_local_offset
                    snapshot.update(
                        {
                            "finger_link": getattr(finger_link, "name", str(finger_link)),
                            "finger_pos": _debug_array_value(finger_pos),
                            "finger_marker_dist": float(th.linalg.norm(finger_pos - marker_pos_for_snapshot)),
                            "finger_marker_local": _debug_array_value(marker_local),
                            "source_residual_marker_local": _debug_array_value(source_residual),
                            "source_residual_norm": float(th.linalg.norm(source_residual)),
                        }
                    )
                return snapshot

            batch_size = env.primitive._motion_generator.batch_size
            target_pos = {}
            target_quat = {}
            prealign_records = []
            for arm_name in active_arms:
                finger_link, finger_pos, finger_records = _select_finger(arm_name)
                if finger_link is None:
                    prealign_records.append({"arm": arm_name, "applied": False, "reason": "no_finger_links"})
                    continue
                eef_link_name = robot.eef_link_names[arm_name]
                eef_pos = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype)
                eef_quat = th.as_tensor(robot.eef_links[arm_name].get_position_orientation()[1], dtype=marker_pos.dtype)
                desired_finger_pos = marker_pos + marker_rot @ marker_local_offset + world_offset
                desired_finger_pos = desired_finger_pos + th.tensor(
                    [0.0, 0.0, approach_z], dtype=desired_finger_pos.dtype, device=desired_finger_pos.device
                )
                full_desired_finger_pos = desired_finger_pos
                if target_fraction < 1.0:
                    desired_finger_pos = finger_pos + target_fraction * (full_desired_finger_pos - finger_pos)
                eef_to_finger_offset = eef_pos - finger_pos
                desired_eef_pos = desired_finger_pos + offset_scale * eef_to_finger_offset
                orientation_mode = (
                    os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ORIENTATION_MODE", "current_eef")
                    or "current_eef"
                ).strip().lower()
                if orientation_mode == "current_eef":
                    desired_eef_quat = eef_quat
                elif orientation_mode == "current_finger":
                    desired_eef_quat = th.as_tensor(finger_link.get_position_orientation()[1], dtype=marker_pos.dtype)
                else:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ORIENTATION_MODE only supports "
                        "current_eef/current_finger"
                    )
                target_pos[eef_link_name] = desired_eef_pos
                target_quat[eef_link_name] = desired_eef_quat
                prealign_records.append(
                    {
                        "arm": arm_name,
                        "applied": True,
                        "eef_link": eef_link_name,
                        "finger_link": getattr(finger_link, "name", str(finger_link)),
                        "finger_body_name": getattr(
                            finger_link,
                            "body_name",
                            str(getattr(finger_link, "name", finger_link)).split(":")[-1],
                        ),
                        "force_finger_link": force_finger_link or None,
                        "all_finger_link_dists": [
                            {
                                "link": getattr(record_finger_link, "name", str(record_finger_link)),
                                "body_name": getattr(
                                    record_finger_link,
                                    "body_name",
                                    str(getattr(record_finger_link, "name", record_finger_link)).split(":")[-1],
                                ),
                                "prim_path": getattr(record_finger_link, "prim_path", None),
                                "dist": record_dist,
                                "selected": record_finger_link is finger_link,
                            }
                            for record_finger_link, _, record_dist in finger_records
                        ],
                        "before": _arm_snapshot(arm_name, finger_link=finger_link),
                        "desired_finger_pos": _debug_array_value(desired_finger_pos),
                        "full_desired_finger_pos": _debug_array_value(full_desired_finger_pos),
                        "desired_eef_pos": _debug_array_value(desired_eef_pos),
                        "desired_eef_quat": _debug_array_value(desired_eef_quat),
                        "eef_to_finger_offset": _debug_array_value(eef_to_finger_offset),
                        "offset_scale": offset_scale,
                        "target_fraction": target_fraction,
                    }
                )

            hold_arms_raw = os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HOLD_ARMS", "right").strip()
            hold_arms = [value.strip() for value in hold_arms_raw.split(",") if value.strip()]
            invalid_hold_arms = [arm_name for arm_name in hold_arms if arm_name not in ("left", "right")]
            if invalid_hold_arms:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HOLD_ARMS must be a comma-separated "
                    f"subset of left,right; got {invalid_hold_arms}"
                )
            hold_records = []
            for arm_name in hold_arms:
                if arm_name in active_arms:
                    continue
                eef_link_name = robot.eef_link_names[arm_name]
                eef_pos, eef_quat = robot.get_eef_pose(arm_name)
                target_pos[eef_link_name] = th.as_tensor(eef_pos, dtype=marker_pos.dtype, device=marker_pos.device)
                target_quat[eef_link_name] = th.as_tensor(eef_quat, dtype=marker_pos.dtype, device=marker_pos.device)
                hold_records.append(_arm_snapshot(arm_name))

            if not target_pos:
                _record(
                    {
                        "applied": False,
                        "reason": "no_targets",
                        "records": prealign_records,
                        "hold_records": hold_records,
                    }
                )
                _set_status(attempted=False, passed=False, reason="no_targets")
                return local_env_step
            _set_status(attempted=True, passed=False, reason="started")

            planning_target_pos = {k: th.stack([v for _ in range(batch_size)]) for k, v in target_pos.items()}
            planning_target_quat = {k: th.stack([v for _ in range(batch_size)]) for k, v in target_quat.items()}
            max_attempts = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_ATTEMPTS", "50") or 50)
            timeout = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_TIMEOUT", "60.0") or 60.0)
            self_collision_check = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_SELF_COLLISION_CHECK", "1") or 1)
            )
            primary_link_override = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_PRIMARY_LINK_OVERRIDE", ""
            ).strip()
            motion_constraint_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MOTION_CONSTRAINT", ""
            ).strip()
            motion_constraint = None
            if motion_constraint_raw:
                motion_constraint_values = [
                    float(value.strip()) for value in motion_constraint_raw.split(",") if value.strip()
                ]
                if len(motion_constraint_values) != 6:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MOTION_CONSTRAINT must contain "
                        "6 comma-separated floats"
                    )
                motion_constraint = th.tensor(motion_constraint_values, dtype=marker_pos.dtype)
            use_attached_obj = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_USE_ATTACHED_OBJ", "1") or 1)
            )
            stale_attached_obj = attached_obj if isinstance(attached_obj, dict) else None
            try:
                stale_attached_obj_scale = attached_obj_scale if isinstance(attached_obj_scale, dict) else None
            except NameError:
                stale_attached_obj_scale = None
            live_attached_obj = None
            live_attached_obj_scale = None
            live_attached_grasp_action = None
            live_attached_error = None
            if use_attached_obj:
                try:
                    live_attached_snapshot = self.obtain_attached_object(env, robot)
                    live_attached_obj = live_attached_snapshot.get("attached_obj")
                    live_attached_obj_scale = live_attached_snapshot.get("attached_obj_scale")
                    live_attached_grasp_action = live_attached_snapshot.get("grasp_action")
                except Exception as e:
                    live_attached_error = f"{type(e).__name__}: {e}"

            attached_for_plan = None
            attached_scale_for_plan = None
            attached_obj_source = "disabled"
            if use_attached_obj:
                if isinstance(live_attached_obj, dict) and live_attached_obj:
                    attached_for_plan = live_attached_obj
                    attached_scale_for_plan = live_attached_obj_scale
                    attached_obj_source = "live"
                elif isinstance(stale_attached_obj, dict) and stale_attached_obj:
                    attached_for_plan = stale_attached_obj
                    attached_scale_for_plan = stale_attached_obj_scale
                    attached_obj_source = "stale_fallback"
                else:
                    attached_obj_source = "none"
            plan_record = {
                "applied": True,
                "stage": "plan_start",
                "marker_pos": _debug_array_value(marker_pos),
                "marker_quat": _debug_array_value(marker_quat),
                "marker_local_offset": _debug_array_value(marker_local_offset),
                "world_offset": _debug_array_value(world_offset),
                "approach_z": approach_z,
                "target_fraction": target_fraction,
                "active_arms": list(active_arms),
                "hold_arms": list(hold_arms),
                "target_pos_by_link": {k: _debug_array_value(v) for k, v in target_pos.items()},
                "target_quat_by_link": {k: _debug_array_value(v) for k, v in target_quat.items()},
                "records": prealign_records,
                "hold_records": hold_records,
                "max_attempts": max_attempts,
                "timeout": timeout,
                "self_collision_check": self_collision_check,
                "primary_link_override": primary_link_override or None,
                "motion_constraint": None if motion_constraint is None else _debug_array_value(motion_constraint),
                "use_attached_obj": use_attached_obj,
                "attached_obj_keys": list((attached_for_plan or {}).keys()) if attached_for_plan else [],
                "attached_obj_source": attached_obj_source,
                "stale_attached_obj_keys": list((stale_attached_obj or {}).keys()) if stale_attached_obj else [],
                "stale_attached_obj_scale": stale_attached_obj_scale or {},
                "live_attached_obj_keys": list((live_attached_obj or {}).keys()) if live_attached_obj else [],
                "live_attached_obj_scale": live_attached_obj_scale or {},
                "live_attached_grasp_action": live_attached_grasp_action,
                "live_attached_error": live_attached_error,
                "attached_obj_options": _attached_payload_options(attached_for_plan),
            }
            _record(plan_record)

            planning_start = time.time()
            old_primary_link_override = os.environ.get("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE")
            if primary_link_override:
                os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = primary_link_override
            try:
                try:
                    mp_results, traj_paths = _compute_trajectories_with_paths(
                        env.cmg,
                        target_pos=planning_target_pos,
                        target_quat=planning_target_quat,
                        is_local=False,
                        max_attempts=max_attempts,
                        timeout=timeout,
                        ik_fail_return=int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_IK_FAIL_RETURN", "10") or 10),
                        enable_finetune_trajopt=bool(
                            int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FINETUNE", "1") or 1)
                        ),
                        finetune_attempts=int(
                            os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FINETUNE_ATTEMPTS", "1") or 1
                        ),
                        return_full_result=True,
                        success_ratio=1.0 / batch_size,
                        attached_obj=attached_for_plan,
                        attached_obj_scale=attached_scale_for_plan,
                        attached_obj_options=_attached_payload_options(attached_for_plan),
                        motion_constraint=motion_constraint,
                        self_collision_check=self_collision_check,
                        emb_sel=emb_sel,
                    )
                finally:
                    if primary_link_override:
                        if old_primary_link_override is None:
                            os.environ.pop("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE", None)
                        else:
                            os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = old_primary_link_override
            except Exception as e:
                _record(
                    {
                        "applied": False,
                        "stage": "plan_exception",
                        "reason": f"{type(e).__name__}: {e}",
                        "planning_time": round(time.time() - planning_start, 2),
                    }
                )
                _set_status(attempted=True, passed=False, reason=f"plan_exception:{type(e).__name__}: {e}")
                return local_env_step

            successes = mp_results[0].success
            success_idx = th.where(successes)[0].cpu()
            status_value = _mp_status_value(mp_results[0])
            _record(
                {
                    "applied": bool(len(success_idx) > 0),
                    "stage": "plan_result",
                    "status": status_value,
                    "success_idx": success_idx.tolist(),
                    "successes": None if _debug_to_np(successes) is None else _debug_to_np(successes).astype(bool).tolist(),
                    "planning_time": round(time.time() - planning_start, 2),
                }
            )
            planned_fk_admission_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_PLANNED_FK_ADMISSION", "1") or 1)
            )
            planned_fk_max_eef_pos_err = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_PLANNED_FK_MAX_EEF_POS_ERR", "0.05")
                or 0.05
            )
            planned_fk_max_finger_target_err = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_FINGER_TARGET_ERR",
                    os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_PLANNED_FK_MAX_FINGER_TARGET_ERR",
                        "0.05",
                    ),
                )
                or 0.05
            )
            def _planned_fk_link_pose(robot_state, link_name):
                link_key = str(link_name).split(":")[-1]
                return robot_state.link_poses.get(link_key) or robot_state.link_poses.get(link_name)

            prealign_quality_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_RANKING", "0") or 0)
            )
            prealign_quality_reject = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_REJECT", "0") or 0)
            )
            prealign_quality_max_base_path = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_MAX_BASE_PATH_M", "0.35")
                or 0.35
            )
            prealign_quality_max_base_yaw_path = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_MAX_BASE_YAW_PATH_RAD",
                    "0.78539816339",
                )
                or 0.78539816339
            )
            prealign_quality_max_trunk_path = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_MAX_TRUNK_PATH", "1.5")
                or 1.5
            )
            prealign_quality_max_arm_path = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_MAX_ARM_PATH", "2.0")
                or 2.0
            )
            prealign_quality_base_weight = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_BASE_WEIGHT", "4.0")
                or 4.0
            )
            prealign_quality_yaw_weight = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_YAW_WEIGHT", "1.0")
                or 1.0
            )
            prealign_quality_trunk_weight = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_TRUNK_WEIGHT", "1.0")
                or 1.0
            )
            prealign_quality_arm_weight = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_CANDIDATE_QUALITY_ARM_WEIGHT", "1.0")
                or 1.0
            )

            if planned_fk_admission_enabled and len(success_idx) > 0:
                admission_record = {
                    "enabled": True,
                    "applied": True,
                    "phase": int(env.execution_phase_ind),
                    "timing_stage": timing_stage,
                    "raw_success_idx": success_idx.tolist(),
                    "accepted_success_idx": success_idx.tolist(),
                    "max_eef_pos_err": planned_fk_max_eef_pos_err,
                    "max_finger_target_err": planned_fk_max_finger_target_err,
                    "target_links": list(target_pos.keys()),
                    "active_finger_links": [
                        record.get("finger_link") for record in prealign_records if record.get("applied")
                    ],
                    "active_finger_body_names": [
                        record.get("finger_body_name") for record in prealign_records if record.get("applied")
                    ],
                    "candidate_quality_ranking": {
                        "enabled": prealign_quality_enabled,
                        "applied": bool(prealign_quality_enabled and is_default_embodiment(emb_sel)),
                        "reject": prealign_quality_reject,
                        "thresholds": {
                            "max_base_path_m": prealign_quality_max_base_path,
                            "max_base_yaw_path_rad": prealign_quality_max_base_yaw_path,
                            "max_trunk_path": prealign_quality_max_trunk_path,
                            "max_arm_path": prealign_quality_max_arm_path,
                        },
                        "weights": {
                            "base": prealign_quality_base_weight,
                            "yaw": prealign_quality_yaw_weight,
                            "trunk": prealign_quality_trunk_weight,
                            "arm": prealign_quality_arm_weight,
                        },
                    },
                    "candidates": [],
                }
                accepted_success_idx = []
                try:
                    for raw_success_idx in success_idx.tolist():
                        path = traj_paths[int(raw_success_idx)]
                        robot_state = env.cmg.mg[emb_sel].kinematics.compute_kinematics(path)
                        candidate_record = {
                            "success_idx": int(raw_success_idx),
                            "per_eef_link": {},
                            "per_active_finger": {},
                            "failed_checks": [],
                            "passed": True,
                        }
                        for link_name, target_pos_value in target_pos.items():
                            planned_link_pose = robot_state.link_poses.get(link_name)
                            planned_pos_np = (
                                None if planned_link_pose is None else _debug_to_np(planned_link_pose.position[-1])
                            )
                            target_pos_np = _debug_to_np(target_pos_value)
                            link_record = {
                                "target_pos": None if target_pos_np is None else target_pos_np.tolist(),
                                "planned_pos": None if planned_pos_np is None else planned_pos_np.tolist(),
                            }
                            if (
                                target_pos_np is None
                                or planned_pos_np is None
                                or not bool(np.isfinite(target_pos_np).all())
                                or not bool(np.isfinite(planned_pos_np).all())
                            ):
                                link_record["error"] = "nonfinite_or_missing_position"
                                candidate_record["failed_checks"].append(f"{link_name}:eef_missing_or_nonfinite")
                            else:
                                pos_err = float(np.linalg.norm(planned_pos_np - target_pos_np))
                                link_record["planned_to_target_pos_dist"] = pos_err
                                if pos_err > planned_fk_max_eef_pos_err:
                                    candidate_record["failed_checks"].append(f"{link_name}:eef_pos_err")
                            candidate_record["per_eef_link"][link_name] = link_record
                        for record in prealign_records:
                            if not record.get("applied"):
                                continue
                            finger_link_name = record.get("finger_link")
                            finger_body_name = record.get("finger_body_name") or str(finger_link_name).split(":")[-1]
                            desired_finger_pos_np = _debug_to_np(record.get("desired_finger_pos", {}).get("value"))
                            planned_finger_pose = _planned_fk_link_pose(robot_state, finger_body_name)
                            planned_finger_pos_np = (
                                None
                                if planned_finger_pose is None
                                else _debug_to_np(planned_finger_pose.position[-1])
                            )
                            finger_record = {
                                "arm": record.get("arm"),
                                "link": finger_link_name,
                                "body_name": finger_body_name,
                                "desired_finger_pos": None
                                if desired_finger_pos_np is None
                                else desired_finger_pos_np.tolist(),
                                "planned_finger_pos": None
                                if planned_finger_pos_np is None
                                else planned_finger_pos_np.tolist(),
                            }
                            if (
                                desired_finger_pos_np is None
                                or planned_finger_pos_np is None
                                or not bool(np.isfinite(desired_finger_pos_np).all())
                                or not bool(np.isfinite(planned_finger_pos_np).all())
                            ):
                                finger_record["error"] = "nonfinite_or_missing_position"
                                candidate_record["failed_checks"].append(
                                    f"{finger_body_name}:finger_missing_or_nonfinite"
                                )
                            else:
                                finger_err = float(np.linalg.norm(planned_finger_pos_np - desired_finger_pos_np))
                                finger_record["planned_to_desired_finger_pos_dist"] = finger_err
                                if finger_err > planned_fk_max_finger_target_err:
                                    candidate_record["failed_checks"].append(f"{finger_body_name}:finger_target_err")
                            candidate_record["per_active_finger"][finger_body_name] = finger_record
                        if prealign_quality_enabled and is_default_embodiment(emb_sel):
                            try:
                                q_path_for_quality = env.cmg.path_to_joint_trajectory(
                                    path,
                                    get_full_js=True,
                                    emb_sel=emb_sel,
                                )
                                quality_by_group = _joint_path_quality_by_group(robot, q_path_for_quality)
                                base_quality = quality_by_group.get("base", {})
                                trunk_quality = quality_by_group.get("trunk", {})
                                arm_left_quality = quality_by_group.get("arm_left", {})
                                arm_right_quality = quality_by_group.get("arm_right", {})
                                quality_failures = []
                                base_path = float(base_quality.get("path_m", 0.0) or 0.0)
                                base_yaw_path = float(base_quality.get("path_rad", 0.0) or 0.0)
                                trunk_path = float(trunk_quality.get("path", 0.0) or 0.0)
                                arm_left_path = float(arm_left_quality.get("path", 0.0) or 0.0)
                                arm_right_path = float(arm_right_quality.get("path", 0.0) or 0.0)
                                if base_path > prealign_quality_max_base_path:
                                    quality_failures.append("base_path")
                                if base_yaw_path > prealign_quality_max_base_yaw_path:
                                    quality_failures.append("base_yaw_path")
                                if trunk_path > prealign_quality_max_trunk_path:
                                    quality_failures.append("trunk_path")
                                if max(arm_left_path, arm_right_path) > prealign_quality_max_arm_path:
                                    quality_failures.append("arm_path")
                                quality_score = (
                                    prealign_quality_base_weight * base_path
                                    + prealign_quality_yaw_weight * base_yaw_path
                                    + prealign_quality_trunk_weight * trunk_path
                                    + prealign_quality_arm_weight * (arm_left_path + arm_right_path)
                                )
                                candidate_record["candidate_quality"] = {
                                    "applied": True,
                                    "score": float(quality_score),
                                    "failures": quality_failures,
                                    "by_group": quality_by_group,
                                }
                                if prealign_quality_reject and quality_failures:
                                    candidate_record["failed_checks"].extend(
                                        [f"candidate_quality_{name}" for name in quality_failures]
                                    )
                            except Exception as exc:
                                candidate_record["candidate_quality"] = {
                                    "applied": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                if prealign_quality_reject:
                                    candidate_record["failed_checks"].append("candidate_quality_error")
                        candidate_record["passed"] = len(candidate_record["failed_checks"]) == 0
                        if candidate_record["passed"]:
                            accepted_success_idx.append(int(raw_success_idx))
                        admission_record["candidates"].append(candidate_record)
                    if prealign_quality_enabled and accepted_success_idx:
                        accepted_candidate_records = [
                            candidate
                            for candidate in admission_record["candidates"]
                            if int(candidate.get("success_idx", -1)) in set(accepted_success_idx)
                        ]
                        accepted_candidate_records.sort(
                            key=lambda candidate: float(
                                candidate.get("candidate_quality", {}).get("score", float("inf"))
                            )
                        )
                        accepted_success_idx = [
                            int(candidate["success_idx"]) for candidate in accepted_candidate_records
                        ]
                        admission_record["candidate_quality_ranking"][
                            "sorted_accepted_success_idx"
                        ] = accepted_success_idx
                    admission_record["accepted_success_idx"] = accepted_success_idx
                    success_idx = th.as_tensor(accepted_success_idx, dtype=th.long)
                except Exception as exc:
                    admission_record["error"] = f"{type(exc).__name__}: {exc}"
                    admission_record["accepted_success_idx"] = []
                    success_idx = th.as_tensor([], dtype=th.long)
                _record(
                    {
                        "applied": bool(len(success_idx) > 0),
                        "stage": "planned_fk_admission",
                        **admission_record,
                    }
                )
            diag_variants_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_DIAG_VARIANTS", ""
            ).strip()
            if len(success_idx) == 0 and diag_variants_raw:
                diag_timeout = float(
                    os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_DIAG_TIMEOUT", "20.0") or 20.0
                )
                diag_max_attempts = int(
                    os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_DIAG_MAX_ATTEMPTS", "20") or 20
                )
                diag_results = []

                def _diag_targets_for_variant(variant_name):
                    variant_target_pos = target_pos
                    variant_target_quat = target_quat
                    if variant_name in ("left_only", "active_only"):
                        keep_links = {robot.eef_link_names[arm_name] for arm_name in active_arms}
                        variant_target_pos = {
                            link_name: value for link_name, value in target_pos.items() if link_name in keep_links
                        }
                        variant_target_quat = {
                            link_name: value for link_name, value in target_quat.items() if link_name in keep_links
                        }
                    return (
                        {k: th.stack([v for _ in range(batch_size)]) for k, v in variant_target_pos.items()},
                        {k: th.stack([v for _ in range(batch_size)]) for k, v in variant_target_quat.items()},
                        list(variant_target_pos.keys()),
                    )

                for diag_variant in [value.strip() for value in diag_variants_raw.split(",") if value.strip()]:
                    diag_target_pos, diag_target_quat, diag_target_keys = _diag_targets_for_variant(diag_variant)
                    diag_attached = attached_for_plan
                    diag_attached_scale = attached_scale_for_plan
                    diag_self_collision = self_collision_check
                    if diag_variant in ("no_attached", "left_only_no_attached"):
                        diag_attached = None
                        diag_attached_scale = None
                    if diag_variant in ("no_self_collision", "left_only_no_self_collision"):
                        diag_self_collision = False
                    if diag_variant == "left_only_no_attached":
                        diag_target_pos, diag_target_quat, diag_target_keys = _diag_targets_for_variant("left_only")
                        diag_attached = None
                        diag_attached_scale = None
                    if diag_variant == "left_only_no_self_collision":
                        diag_target_pos, diag_target_quat, diag_target_keys = _diag_targets_for_variant("left_only")
                        diag_self_collision = False
                    diag_start = time.time()
                    old_diag_primary_link_override = os.environ.get("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE")
                    if primary_link_override:
                        os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = primary_link_override
                    try:
                        try:
                            diag_mp_results, _ = _compute_trajectories_with_paths(
                                env.cmg,
                                target_pos=diag_target_pos,
                                target_quat=diag_target_quat,
                                is_local=False,
                                max_attempts=diag_max_attempts,
                                timeout=diag_timeout,
                                ik_fail_return=int(
                                    os.environ.get(
                                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_IK_FAIL_RETURN", "10"
                                    )
                                    or 10
                                ),
                                enable_finetune_trajopt=bool(
                                    int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FINETUNE", "1") or 1)
                                ),
                                finetune_attempts=int(
                                    os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FINETUNE_ATTEMPTS", "1")
                                    or 1
                                ),
                                return_full_result=True,
                                success_ratio=1.0 / batch_size,
                                attached_obj=diag_attached,
                                attached_obj_scale=diag_attached_scale,
                                attached_obj_options=_attached_payload_options(diag_attached),
                                motion_constraint=motion_constraint,
                                self_collision_check=diag_self_collision,
                                emb_sel=emb_sel,
                            )
                            diag_successes = diag_mp_results[0].success
                            diag_success_idx = th.where(diag_successes)[0].cpu()
                            diag_status_obj = getattr(diag_mp_results[0], "status", None)
                            diag_status_value = getattr(diag_status_obj, "value", str(diag_status_obj))
                            diag_results.append(
                                {
                                    "variant": diag_variant,
                                    "applied": bool(len(diag_success_idx) > 0),
                                    "status": diag_status_value,
                                    "success_idx": diag_success_idx.tolist(),
                                    "successes": None
                                    if _debug_to_np(diag_successes) is None
                                    else _debug_to_np(diag_successes).astype(bool).tolist(),
                                    "planning_time": round(time.time() - diag_start, 2),
                                    "target_keys": diag_target_keys,
                                    "attached_obj_keys": list((diag_attached or {}).keys()) if diag_attached else [],
                                    "self_collision_check": bool(diag_self_collision),
                                }
                            )
                        finally:
                            if primary_link_override:
                                if old_diag_primary_link_override is None:
                                    os.environ.pop("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE", None)
                                else:
                                    os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = old_diag_primary_link_override
                    except Exception as e:
                        diag_results.append(
                            {
                                "variant": diag_variant,
                                "applied": False,
                                "status": "exception",
                                "reason": f"{type(e).__name__}: {e}",
                                "planning_time": round(time.time() - diag_start, 2),
                                "target_keys": diag_target_keys,
                                "attached_obj_keys": list((diag_attached or {}).keys()) if diag_attached else [],
                                "self_collision_check": bool(diag_self_collision),
                            }
                        )
                _record(
                    {
                        "applied": False,
                        "stage": "plan_diag_variants",
                        "diag_timeout": diag_timeout,
                        "diag_max_attempts": diag_max_attempts,
                        "results": diag_results,
                    }
                )
            if len(success_idx) == 0:
                _set_status(attempted=True, passed=False, reason="no_admitted_plan")
                return local_env_step

            q_traj = env.cmg.path_to_joint_trajectory(
                traj_paths[success_idx[0]],
                get_full_js=True,
                emb_sel=emb_sel,
            )
            q_traj = _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01)
            max_q_traj_steps = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_Q_TRAJ_STEPS", "0") or 0)
            if max_q_traj_steps > 0 and q_traj.shape[0] > max_q_traj_steps:
                keep_idx = th.linspace(0, q_traj.shape[0] - 1, max_q_traj_steps, device=q_traj.device).long()
                q_traj = q_traj[keep_idx]
            q_traj = q_traj.cpu()

            active_gripper_action_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ACTIVE_GRIPPER_ACTION",
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_ACTIVE_GRIPPER_ACTION", ""),
            ).strip()
            active_gripper_action = None if active_gripper_action_raw == "" else float(active_gripper_action_raw)
            hold_left_gripper_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HOLD_LEFT_GRIPPER_ACTION",
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_HOLD_LEFT_GRIPPER_ACTION", ""),
            ).strip()
            hold_right_gripper_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HOLD_RIGHT_GRIPPER_ACTION",
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_HOLD_RIGHT_GRIPPER_ACTION", ""),
            ).strip()
            hold_left_gripper_action = None if hold_left_gripper_raw == "" else float(hold_left_gripper_raw)
            hold_right_gripper_action = None if hold_right_gripper_raw == "" else float(hold_right_gripper_raw)

            def _apply_prealign_grippers(action):
                if left_gripper_action is not None:
                    left_cmd = left_gripper_action[0]
                    if active_gripper_action is not None and "left" in active_arms:
                        left_cmd = active_gripper_action
                    if hold_left_gripper_action is not None:
                        left_cmd = hold_left_gripper_action
                    action[env_interface.gripper_action_dim[0]] = left_cmd
                if right_gripper_action is not None:
                    right_cmd = right_gripper_action[1]
                    if active_gripper_action is not None and "right" in active_arms:
                        right_cmd = active_gripper_action
                    if hold_right_gripper_action is not None:
                        right_cmd = hold_right_gripper_action
                    action[env_interface.gripper_action_dim[1]] = right_cmd
                return action

            execution_start = time.time()
            exec_debug_interval = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_EXEC_DEBUG_INTERVAL", "50") or 50
            )
            for q_idx, q_point in enumerate(q_traj):
                mp_action = _joint_trajectory_point_to_action(robot, q_point.detach().clone()).cpu().numpy()
                mp_action = _apply_prealign_grippers(mp_action)
                state = env.get_state()["states"]
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=mp_action)
                mp_action = _postprocess_action_compatible(env, mp_action)
                env.step(mp_action, video_writer)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(mp_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]
                if exec_debug_interval > 0 and (q_idx == 0 or q_idx % exec_debug_interval == 0):
                    _record(
                        {
                            "applied": True,
                            "stage": "exec_step",
                            "q_idx": int(q_idx),
                            "q_traj_len": int(q_traj.shape[0]),
                            "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                        }
                    )

            try:
                fresh_marker_pos_raw, fresh_marker_quat_raw = marker.get_position_orientation()
                fresh_marker_pos = th.as_tensor(fresh_marker_pos_raw, dtype=marker_pos.dtype)
                fresh_marker_quat = th.as_tensor(fresh_marker_quat_raw, dtype=marker_pos.dtype)
                fresh_marker_rot = th.as_tensor(T.quat2mat(fresh_marker_quat), dtype=marker_pos.dtype)
                fresh_ref_obj_pos_raw, fresh_ref_obj_quat_raw = ref_obj.get_position_orientation()
                fresh_ref_obj_pos = th.as_tensor(fresh_ref_obj_pos_raw, dtype=marker_pos.dtype)
                fresh_ref_obj_quat = th.as_tensor(fresh_ref_obj_quat_raw, dtype=marker_pos.dtype)
                fresh_marker_error = None
            except Exception as e:
                fresh_marker_pos = marker_pos
                fresh_marker_quat = marker_quat
                fresh_marker_rot = marker_rot
                fresh_ref_obj_pos = None
                fresh_ref_obj_quat = None
                fresh_marker_error = f"{type(e).__name__}: {e}"
            after_records = []
            hard_validation_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HARD_VALIDATION", "1") or 1)
            )
            hard_validation_max_finger_target_err = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HARD_VALIDATION_MAX_FINGER_TARGET_ERR",
                    os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_FINGER_TARGET_ERR", "0.05"),
                )
                or 0.05
            )
            hard_validation_record = {
                "enabled": hard_validation_enabled,
                "passed": True,
                "max_finger_target_err": hard_validation_max_finger_target_err,
                "per_active_finger": {},
            }
            for record in prealign_records:
                if not record.get("applied"):
                    continue
                arm_name = record["arm"]
                finger_link, _, _ = _select_finger(arm_name)
                stale_snapshot = _arm_snapshot(arm_name, finger_link=finger_link)
                fresh_snapshot = _arm_snapshot(
                    arm_name,
                    finger_link=finger_link,
                    snapshot_marker_pos=fresh_marker_pos,
                    snapshot_marker_rot=fresh_marker_rot,
                )
                fresh_snapshot["stale_marker_snapshot"] = stale_snapshot
                after_records.append(fresh_snapshot)
                if hard_validation_enabled:
                    fresh_finger_pos = th.as_tensor(
                        finger_link.get_position_orientation()[0], dtype=fresh_marker_pos.dtype
                    )
                    fresh_desired_finger_pos = fresh_marker_pos + fresh_marker_rot @ marker_local_offset + world_offset
                    fresh_desired_finger_pos = fresh_desired_finger_pos + th.tensor(
                        [0.0, 0.0, approach_z],
                        dtype=fresh_desired_finger_pos.dtype,
                        device=fresh_desired_finger_pos.device,
                    )
                    finger_target_err = float(th.linalg.norm(fresh_finger_pos - fresh_desired_finger_pos))
                    finger_passed = finger_target_err <= hard_validation_max_finger_target_err
                    hard_validation_record["per_active_finger"][getattr(finger_link, "name", str(finger_link))] = {
                        "arm": arm_name,
                        "finger_pos": _debug_array_value(fresh_finger_pos),
                        "fresh_desired_finger_pos": _debug_array_value(fresh_desired_finger_pos),
                        "finger_to_fresh_target_pos_dist": finger_target_err,
                        "passed": bool(finger_passed),
                    }
                    if not finger_passed:
                        hard_validation_record["passed"] = False
            if not hard_validation_enabled:
                hard_validation_record["passed"] = None
            _record(
                {
                    "applied": bool(hard_validation_record["passed"] is not False),
                    "stage": "exec_done",
                    "q_traj_len": int(q_traj.shape[0]),
                    "execution_time": round(time.time() - execution_start, 2),
                    "plan_marker_pos": _debug_array_value(marker_pos),
                    "plan_marker_quat": _debug_array_value(marker_quat),
                    "fresh_marker_pos": _debug_array_value(fresh_marker_pos),
                    "fresh_marker_quat": _debug_array_value(fresh_marker_quat),
                    "fresh_marker_delta_world": _debug_array_value(fresh_marker_pos - marker_pos),
                    "fresh_marker_drift_from_plan": float(th.linalg.norm(fresh_marker_pos - marker_pos)),
                    "fresh_ref_obj_pos": _debug_array_value(fresh_ref_obj_pos),
                    "fresh_ref_obj_quat": _debug_array_value(fresh_ref_obj_quat),
                    "fresh_marker_error": fresh_marker_error,
                    "after_records": after_records,
                    "hard_validation": hard_validation_record,
                    "success": {k: bool(v) for k, v in success.items()},
                }
            )
            if hard_validation_record["passed"] is False:
                _set_status(attempted=True, passed=False, reason="hard_validation_failed")
            else:
                _set_status(attempted=True, passed=True, reason="passed")
            return local_env_step

        def _maybe_execute_toggle_marker_post_mp_press(
            left_gripper_action,
            right_gripper_action,
            video_writer,
            states,
            actions,
            observations,
            observations_info,
            datagen_infos,
            success,
            local_env_step,
            execute_live_q_to_action,
            wholebody_arm_mp_enabled,
            emb_sel,
            press_timing_stage="post_mp",
        ):
            """Env-gated diagnostic press after whole-body MP reaches the carried live marker."""
            press_steps = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_STEPS", "0") or 0)
            if press_steps <= 0:
                return local_env_step
            if not (
                execute_live_q_to_action
                and wholebody_arm_mp_enabled
                and is_default_embodiment(emb_sel)
            ):
                return local_env_step
            min_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_MIN_PHASE", "0") or 0)
            max_phase = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_MAX_PHASE", "999999") or 999999)
            if not (min_phase <= int(env.execution_phase_ind) <= max_phase):
                return local_env_step
            if ref_obj is None or object_states.ToggledOn not in getattr(ref_obj, "states", {}):
                return local_env_step

            log_key = "toggle_marker_post_mp_press"
            desired_press_timing = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_TIMING", "post_mp"
            ).strip().lower()
            valid_press_timings = {"post_mp", "after_arm_replay"}
            if desired_press_timing not in valid_press_timings:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_TIMING must be one of "
                    f"{sorted(valid_press_timings)}, got {desired_press_timing!r}"
                )
            if desired_press_timing != press_timing_stage:
                return local_env_step

            def _record(record):
                record.setdefault("timing_stage", press_timing_stage)
                record.setdefault("press_timing", desired_press_timing)
                phase_logs.setdefault(env.execution_phase_ind, {}).setdefault(log_key, []).append(record)
                print("[MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS] " + json.dumps(record, default=str), flush=True)

            ref_name = getattr(ref_obj, "name", None)
            active_arms = []
            for arm_name in ("left", "right"):
                arm_ref = object_ref.get(f"arm_{arm_name}")
                if arm_ref is ref_obj or getattr(arm_ref, "name", None) == ref_name or str(arm_ref) == str(ref_name):
                    active_arms.append(arm_name)
            active_arms_override_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_ACTIVE_ARMS_OVERRIDE", ""
            ).strip()
            if active_arms_override_raw:
                requested_active_arms = [
                    value.strip()
                    for value in active_arms_override_raw.split(",")
                    if value.strip()
                ]
                invalid_active_arms = [
                    arm_name for arm_name in requested_active_arms if arm_name not in ("left", "right")
                ]
                if invalid_active_arms:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_ACTIVE_ARMS_OVERRIDE only supports "
                        f"'left' and/or 'right', got {invalid_active_arms}"
                    )
                _record(
                    {
                        "enabled": True,
                        "applied": True,
                        "phase": int(env.execution_phase_ind),
                        "reason": "active_arms_override",
                        "object_ref_active_arms": list(active_arms),
                        "requested_active_arms": requested_active_arms,
                    }
                )
                active_arms = requested_active_arms
            if not active_arms:
                _record(
                    {
                        "enabled": True,
                        "applied": False,
                        "phase": int(env.execution_phase_ind),
                        "reason": "no_active_toggle_arm",
                    }
                )
                return local_env_step

            finger_servo_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_SERVO", "0") or 0)
            )
            offset_scale = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_OFFSET_SCALE", "1.0") or 1.0)
            approach_z = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_APPROACH_Z", "0.0") or 0.0)
            marker_local_offset_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_LOCAL_OFFSET", ""
            ).strip()
            marker_local_offset = None
            if marker_local_offset_raw:
                marker_local_offset_values = [
                    float(value.strip()) for value in marker_local_offset_raw.split(",") if value.strip()
                ]
                if len(marker_local_offset_values) != 3:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_LOCAL_OFFSET must contain 3 comma-separated floats"
                    )
                marker_local_offset = th.tensor(marker_local_offset_values, dtype=th.float32)
            world_offset_raw = os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_WORLD_OFFSET", "").strip()
            world_offset = None
            if world_offset_raw:
                world_offset_values = [
                    float(value.strip()) for value in world_offset_raw.split(",") if value.strip()
                ]
                if len(world_offset_values) != 3:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_WORLD_OFFSET must contain 3 comma-separated floats"
                    )
                world_offset = th.tensor(world_offset_values, dtype=th.float32)
            residual_comp_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_RESIDUAL_COMP_GAIN", "0.0") or 0.0
            )
            residual_comp_max_norm = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_RESIDUAL_COMP_MAX_NORM", "0.0") or 0.0
            )
            residual_comp_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_RESIDUAL_COMP_AXES", "1,1,1"
            ).strip()
            residual_comp_axes_values = [
                float(value.strip()) for value in residual_comp_axes_raw.split(",") if value.strip()
            ]
            if len(residual_comp_axes_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_RESIDUAL_COMP_AXES must contain 3 comma-separated floats")
            residual_comp_axes = th.tensor(residual_comp_axes_values, dtype=th.float32)
            log_interval = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS_LOG_INTERVAL", "1") or 1)
            max_step = float(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MAX_STEP", "0.0") or 0.0)
            convergence_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONVERGENCE_THRESHOLD", "0.0") or 0.0
            )
            marker_drift_limit = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_DRIFT_LIMIT", "0.0") or 0.0
            )
            no_progress_steps = int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_NO_PROGRESS_STEPS", "0") or 0)
            no_progress_epsilon = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_NO_PROGRESS_EPS", "1e-4") or 1e-4
            )
            no_progress_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_NO_PROGRESS_MIN_STEP", "0") or 0
            )
            no_progress_activate_on_first_hit = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_NO_PROGRESS_AFTER_FIRST_HIT", "0") or 0)
            )
            post_step_regress_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_POST_STEP_REGRESS_STOP", "0") or 0)
            )
            post_step_regress_steps = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_POST_STEP_REGRESS_STEPS", "0") or 0
            )
            post_step_regress_epsilon = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_POST_STEP_REGRESS_EPS", "1e-4") or 1e-4
            )
            post_step_regress_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_POST_STEP_REGRESS_MIN_STEP", "0") or 0
            )
            post_step_regress_activation_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_POST_STEP_REGRESS_ACTIVATION_THRESHOLD", "0.0")
                or 0.0
            )
            progress_metric = (
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PROGRESS_METRIC", "finger_dist")
                or "finger_dist"
            ).strip()
            if progress_metric not in {"finger_dist", "overlap_radius"}:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_PROGRESS_METRIC must be one of "
                    "finger_dist, overlap_radius"
                )
            overlap_active_arms_raw = (
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_OVERLAP_ACTIVE_ARMS", "active")
                or "active"
            ).strip()
            if overlap_active_arms_raw == "active":
                overlap_active_arms = list(active_arms)
            else:
                overlap_active_arms = [
                    value.strip()
                    for value in overlap_active_arms_raw.split(",")
                    if value.strip()
                ]
            invalid_overlap_active_arms = [
                arm_name for arm_name in overlap_active_arms if arm_name not in ("left", "right")
            ]
            if invalid_overlap_active_arms:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_OVERLAP_ACTIVE_ARMS must be 'active' "
                    f"or a comma-separated subset of left,right; got {invalid_overlap_active_arms}"
                )
            contact_seek_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK", "0") or 0)
            )
            contact_seek_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_MIN_STEP", "0") or 0
            )
            contact_seek_target_radius = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_TARGET_RADIUS", "0.0") or 0.0
            )
            contact_seek_extra_depth = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_EXTRA_DEPTH", "0.0") or 0.0
            )
            contact_seek_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_AXES", "1,1,1"
            ).strip()
            contact_seek_axes_values = [
                float(value.strip()) for value in contact_seek_axes_raw.split(",") if value.strip()
            ]
            if len(contact_seek_axes_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_AXES must contain 3 comma-separated floats")
            contact_seek_axes = th.tensor(contact_seek_axes_values, dtype=th.float32)
            contact_seek_dir_local_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_DIR_LOCAL", ""
            ).strip()
            contact_seek_dir_local = None
            if contact_seek_dir_local_raw:
                contact_seek_dir_local_values = [
                    float(value.strip()) for value in contact_seek_dir_local_raw.split(",") if value.strip()
                ]
                if len(contact_seek_dir_local_values) != 3:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_DIR_LOCAL must contain 3 comma-separated floats"
                    )
                contact_seek_dir_local = th.tensor(contact_seek_dir_local_values, dtype=th.float32)
            contact_seek_drift_scale_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_DRIFT_SCALE_THRESHOLD", "0.0") or 0.0
            )
            contact_seek_drift_stop_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_DRIFT_STOP_THRESHOLD", "0.0") or 0.0
            )
            contact_seek_drift_min_scale = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_SEEK_DRIFT_MIN_SCALE", "0.0") or 0.0
            )
            active_gripper_action_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_ACTIVE_GRIPPER_ACTION", ""
            ).strip()
            active_gripper_action = (
                None if active_gripper_action_raw == "" else float(active_gripper_action_raw)
            )
            pre_gripper_action_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_PRE_GRIPPER_ACTION", ""
            ).strip()
            pre_gripper_action = (
                active_gripper_action
                if pre_gripper_action_raw == ""
                else float(pre_gripper_action_raw)
            )
            force_finger_link = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_FORCE_FINGER_LINK", ""
            ).strip()
            micro_servo_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO", "0") or 0)
            )
            micro_servo_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO_GAIN", "1.0") or 1.0
            )
            micro_servo_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO_AXES", "1,1,1"
            ).strip()
            micro_servo_axes_values = [
                float(value.strip()) for value in micro_servo_axes_raw.split(",") if value.strip()
            ]
            if len(micro_servo_axes_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO_AXES must contain 3 comma-separated floats")
            micro_servo_axes = th.tensor(micro_servo_axes_values, dtype=th.float32)
            micro_servo_bias_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO_BIAS", "0,0,0"
            ).strip()
            micro_servo_bias_values = [
                float(value.strip()) for value in micro_servo_bias_raw.split(",") if value.strip()
            ]
            if len(micro_servo_bias_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_MICRO_SERVO_BIAS must contain 3 comma-separated floats")
            micro_servo_bias = th.tensor(micro_servo_bias_values, dtype=th.float32)
            finger_jacobian_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN", "0") or 0)
            )
            finger_jacobian_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_GAIN", "1.0") or 1.0
            )
            finger_jacobian_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_AXES", "1,1,1"
            ).strip()
            finger_jacobian_axes_values = [
                float(value.strip()) for value in finger_jacobian_axes_raw.split(",") if value.strip()
            ]
            if len(finger_jacobian_axes_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_AXES must contain 3 comma-separated floats")
            finger_jacobian_axes = th.tensor(finger_jacobian_axes_values, dtype=th.float32)
            finger_jacobian_bias_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_BIAS", "0,0,0"
            ).strip()
            finger_jacobian_bias_values = [
                float(value.strip()) for value in finger_jacobian_bias_raw.split(",") if value.strip()
            ]
            if len(finger_jacobian_bias_values) != 3:
                raise ValueError("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_BIAS must contain 3 comma-separated floats")
            finger_jacobian_bias = th.tensor(finger_jacobian_bias_values, dtype=th.float32)
            finger_jacobian_contact_point_offset_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_CONTACT_POINT_OFFSET", "0,0,0"
            ).strip()
            finger_jacobian_contact_point_offset_values = [
                float(value.strip())
                for value in finger_jacobian_contact_point_offset_raw.split(",")
                if value.strip()
            ]
            if len(finger_jacobian_contact_point_offset_values) != 3:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_CONTACT_POINT_OFFSET "
                    "must contain 3 comma-separated floats"
                )
            finger_jacobian_contact_point_offset = th.tensor(
                finger_jacobian_contact_point_offset_values, dtype=th.float32
            )
            finger_jacobian_step_max_norm = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_STEP_MAX_NORM", "0.02") or 0.02
            )
            finger_jacobian_joint_delta_max_abs = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_JOINT_DELTA_MAX_ABS", "0.04") or 0.04
            )
            finger_jacobian_damping = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_DAMPING", "0.01") or 0.01
            )
            finger_jacobian_joint_limit_margin = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_JOINT_LIMIT_MARGIN", "0.02") or 0.02
            )
            finger_jacobian_include_trunk = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_INCLUDE_TRUNK", "1") or 1)
            )
            finger_jacobian_include_base = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_INCLUDE_BASE", "0") or 0)
            )
            finger_jacobian_base_step_max_norm = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_FINGER_JACOBIAN_BASE_STEP_MAX_NORM", "0.0") or 0.0
            )
            marker_frame_servo_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO", "0") or 0)
            )
            marker_frame_servo_lateral_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_LATERAL_GAIN", "1.0") or 1.0
            )
            marker_frame_servo_normal_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_NORMAL_GAIN", "1.0") or 1.0
            )
            marker_frame_servo_lateral_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_LATERAL_AXES", "1,1"
            ).strip()
            marker_frame_servo_lateral_axes_values = [
                float(value.strip()) for value in marker_frame_servo_lateral_axes_raw.split(",") if value.strip()
            ]
            if len(marker_frame_servo_lateral_axes_values) != 2:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_LATERAL_AXES "
                    "must contain 2 comma-separated floats"
                )
            marker_frame_servo_lateral_axes = th.tensor(marker_frame_servo_lateral_axes_values, dtype=th.float32)
            marker_frame_servo_lateral_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_LATERAL_THRESHOLD", "0.03") or 0.03
            )
            marker_frame_servo_y_staging_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_Y_STAGING", "0") or 0)
            )
            marker_frame_servo_y_staging_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_Y_STAGING_THRESHOLD", "0.04") or 0.04
            )
            marker_frame_servo_y_staging_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_Y_STAGING_MIN_STEP", "0") or 0
            )
            marker_frame_servo_y_staging_max_residual = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_Y_STAGING_MAX_RESIDUAL", "0.0") or 0.0
            )
            marker_frame_servo_object_drift_comp_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_FRAME_SERVO_OBJECT_DRIFT_COMP", "0") or 0)
            )
            coupled_budget_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET", "0") or 0)
            )
            coupled_budget_residual_window = max(
                1,
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET_RESIDUAL_WINDOW", "5") or 5),
            )
            coupled_budget_residual_eps = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET_RESIDUAL_EPS", "0.002") or 0.002
            )
            coupled_budget_scale = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET_SCALE", "0.5") or 0.5
            )
            coupled_budget_min_scale = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET_MIN_SCALE", "0.25") or 0.25
            )
            coupled_budget_ref_obj_drift_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_COUPLED_BUDGET_REF_OBJ_DRIFT_THRESHOLD", "0.20")
                or 0.20
            )
            contact_gate_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE", "0") or 0)
            )
            contact_gate_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_MIN_STEP", "0") or 0
            )
            contact_gate_residual_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_RESIDUAL_THRESHOLD", "0.0") or 0.0
            )
            contact_gate_ref_obj_drift_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_REF_OBJ_DRIFT_THRESHOLD", "0.0")
                or 0.0
            )
            contact_gate_marker_z_lift_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_MARKER_Z_LIFT_THRESHOLD", "0.0")
                or 0.0
            )
            contact_gate_ref_obj_z_lift_threshold = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_REF_OBJ_Z_LIFT_THRESHOLD", "0.0")
                or 0.0
            )
            contact_gate_require_residual_stale = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_GATE_REQUIRE_RESIDUAL_STALE", "0") or 0)
            )
            contact_aware_press_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_PRESS", "0") or 0)
            )
            contact_aware_press_min_step = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_MIN_STEP", "0") or 0
            )
            contact_aware_press_depth = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_DEPTH", "0.006") or 0.006
            )
            contact_aware_allow_precontact = bool(
                int(
                    os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_ALLOW_PRECONTACT", "0"
                    )
                    or 0
                )
            )
            contact_aware_precontact_max_dist = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_PRECONTACT_MAX_DIST", "0.0"
                )
                or 0.0
            )
            contact_aware_precontact_shell_target = bool(
                int(
                    os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_PRECONTACT_SHELL_TARGET", "0"
                    )
                    or 0
                )
            )
            contact_aware_precontact_shell_radius = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_PRECONTACT_SHELL_RADIUS", "0.0"
                )
                or 0.0
            )
            contact_aware_precontact_shell_depth = float(
                os.environ.get(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_PRECONTACT_SHELL_DEPTH", "0.0"
                )
                or 0.0
            )
            contact_aware_press_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_AXES", "1,1,1"
            ).strip()
            contact_aware_press_axes_values = [
                float(value.strip()) for value in contact_aware_press_axes_raw.split(",") if value.strip()
            ]
            if len(contact_aware_press_axes_values) != 3:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_AXES must contain 3 comma-separated floats"
                )
            contact_aware_press_axes = th.tensor(contact_aware_press_axes_values, dtype=th.float32)
            contact_aware_press_dir_local_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_DIR_LOCAL", ""
            ).strip()
            contact_aware_press_dir_local = None
            if contact_aware_press_dir_local_raw:
                contact_aware_press_dir_local_values = [
                    float(value.strip()) for value in contact_aware_press_dir_local_raw.split(",") if value.strip()
                ]
                if len(contact_aware_press_dir_local_values) != 3:
                    raise ValueError(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_DIR_LOCAL must contain 3 comma-separated floats"
                    )
                contact_aware_press_dir_local = th.tensor(contact_aware_press_dir_local_values, dtype=th.float32)
            contact_aware_disable_regress_stop = bool(
                int(
                    os.environ.get(
                        "MOMAGEN_TOGGLE_MARKER_POST_MP_CONTACT_AWARE_DISABLE_REGRESS_STOP", "1"
                    )
                    or 1
                )
            )
            predicate_hold_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PREDICATE_HOLD", "0") or 0)
            )
            predicate_hold_target_steps = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PREDICATE_HOLD_TARGET_STEPS", "5") or 5
            )
            preserve_ref_obj_pose_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_POSE", "0") or 0)
            )
            preserve_ref_obj_pose_gain = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_GAIN", "1.0") or 1.0
            )
            preserve_ref_obj_pose_max_step = float(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_MAX_STEP", "0.0") or 0.0
            )
            preserve_ref_obj_pose_axes_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_AXES", "1,1,1"
            ).strip()
            preserve_ref_obj_pose_axes_values = [
                float(value.strip()) for value in preserve_ref_obj_pose_axes_raw.split(",") if value.strip()
            ]
            if len(preserve_ref_obj_pose_axes_values) != 3:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_AXES must contain 3 comma-separated floats"
                )
            preserve_ref_obj_pose_axes = th.tensor(preserve_ref_obj_pose_axes_values, dtype=th.float32)
            preserve_ref_obj_pose_arms_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_ARMS", "right"
            ).strip()
            preserve_ref_obj_pose_arms = [
                value.strip()
                for value in preserve_ref_obj_pose_arms_raw.split(",")
                if value.strip()
            ]
            invalid_preserve_ref_obj_pose_arms = [
                arm_name for arm_name in preserve_ref_obj_pose_arms if arm_name not in ("left", "right")
            ]
            if invalid_preserve_ref_obj_pose_arms:
                raise ValueError(
                    "MOMAGEN_TOGGLE_MARKER_POST_MP_PRESERVE_REF_OBJ_ARMS must be a comma-separated "
                    f"subset of left,right; got {invalid_preserve_ref_obj_pose_arms}"
                )
            overlap_diag_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_OVERLAP_DIAG", "0") or 0)
            )
            overlap_diag_max_hits = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_OVERLAP_DIAG_MAX_HITS", "32") or 32
            )
            overlap_diag_radii_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_OVERLAP_DIAG_RADII", ""
            ).strip()
            overlap_diag_radii = []
            if overlap_diag_radii_raw:
                overlap_diag_radii = [
                    float(value.strip()) for value in overlap_diag_radii_raw.split(",") if value.strip()
                ]
            neighborhood_diag_enabled = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_NEIGHBORHOOD_DIAG", "0") or 0)
            )
            neighborhood_diag_offsets_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_NEIGHBORHOOD_DIAG_OFFSETS", "0.0,0.02235804684460163,0.05"
            ).strip()
            neighborhood_diag_offsets = [
                float(value.strip()) for value in neighborhood_diag_offsets_raw.split(",") if value.strip()
            ]
            neighborhood_diag_radius_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_NEIGHBORHOOD_DIAG_RADIUS", ""
            ).strip()
            neighborhood_diag_dirs_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_NEIGHBORHOOD_DIAG_DIRS",
                "1,0,0;-1,0,0;0,1,0;0,-1,0;0,0,1;0,0,-1;1,1,0;-1,-1,0;1,-1,0;-1,1,0",
            ).strip()
            neighborhood_diag_dirs = []
            if neighborhood_diag_dirs_raw:
                for item in neighborhood_diag_dirs_raw.split(";"):
                    values = [float(value.strip()) for value in item.split(",") if value.strip()]
                    if len(values) != 3:
                        raise ValueError(
                            "MOMAGEN_TOGGLE_MARKER_POST_MP_NEIGHBORHOOD_DIAG_DIRS must contain "
                            "semicolon-separated 3D vectors"
                        )
                    neighborhood_diag_dirs.append(values)
            pre_gripper_steps = int(
                os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_PRE_GRIPPER_STEPS", "0") or 0
            )
            log_only = bool(
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_LOG_ONLY", "0") or 0)
            )
            action_repeat_steps = max(
                1,
                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_POST_MP_ACTION_REPEAT_STEPS", "1") or 1),
            )
            hold_left_gripper_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_HOLD_LEFT_GRIPPER_ACTION", ""
            ).strip()
            hold_left_gripper_action = (
                None if hold_left_gripper_raw == "" else float(hold_left_gripper_raw)
            )
            hold_right_gripper_raw = os.environ.get(
                "MOMAGEN_TOGGLE_MARKER_POST_MP_HOLD_RIGHT_GRIPPER_ACTION", ""
            ).strip()
            hold_right_gripper_action = (
                None if hold_right_gripper_raw == "" else float(hold_right_gripper_raw)
            )

            def _apply_post_mp_gripper_action(action, gripper_action_override=None):
                gripper_action = active_gripper_action if gripper_action_override is None else gripper_action_override
                if left_gripper_action is not None:
                    left_cmd = left_gripper_action[0]
                    if gripper_action is not None and "left" in active_arms:
                        left_cmd = gripper_action
                    if hold_left_gripper_action is not None:
                        left_cmd = hold_left_gripper_action
                    action[env_interface.gripper_action_dim[0]] = left_cmd
                if right_gripper_action is not None:
                    right_cmd = right_gripper_action[1]
                    if gripper_action is not None and "right" in active_arms:
                        right_cmd = gripper_action
                    if hold_right_gripper_action is not None:
                        right_cmd = hold_right_gripper_action
                    action[env_interface.gripper_action_dim[1]] = right_cmd
                return action

            def _action_slice_diag(action, controller_name, controller):
                try:
                    action_np = np.asarray(_debug_to_np(action), dtype=float)
                    idx = robot.controller_action_idx.get(controller_name)
                    if idx is None:
                        return {"controller": controller_name, "reason": "missing_action_idx"}
                    idx_np = np.asarray(_debug_to_np(idx), dtype=int)
                    values = np.asarray(action_np[idx_np], dtype=float)
                    diag = {
                        "controller": controller_name,
                        "idx": idx_np.tolist(),
                        "values": values.tolist(),
                        "min": float(values.min()) if values.size else None,
                        "max": float(values.max()) if values.size else None,
                        "norm": float(np.linalg.norm(values)),
                    }
                    command_input_limits = getattr(controller, "command_input_limits", None)
                    if command_input_limits is not None:
                        low = np.asarray(_debug_to_np(command_input_limits[0]), dtype=float)
                        high = np.asarray(_debug_to_np(command_input_limits[1]), dtype=float)
                        diag["command_input_low"] = low.tolist()
                        diag["command_input_high"] = high.tolist()
                        diag["outside_command_input_count"] = int(
                            np.sum((values < low - 1e-6) | (values > high + 1e-6))
                        )
                    try:
                        low = np.asarray(_debug_to_np(robot.action_space.low), dtype=float)[idx_np]
                        high = np.asarray(_debug_to_np(robot.action_space.high), dtype=float)[idx_np]
                        diag["action_space_low"] = low.tolist()
                        diag["action_space_high"] = high.tolist()
                        diag["outside_action_space_count"] = int(
                            np.sum((values < low - 1e-6) | (values > high + 1e-6))
                        )
                    except Exception as e:
                        diag["action_space_error"] = f"{type(e).__name__}: {e}"
                    return diag
                except Exception as e:
                    return {"controller": controller_name, "error": f"{type(e).__name__}: {e}"}

            def _record_post_mp_action_diag(action, step_record, arm_name, prefix):
                arm_controller = robot.controllers.get(f"arm_{arm_name}")
                if arm_controller is not None:
                    step_record[f"{prefix}_arm_action_diag"] = _action_slice_diag(
                        action, f"arm_{arm_name}", arm_controller
                    )
                if finger_jacobian_include_trunk:
                    trunk_controller = robot.controllers.get("trunk")
                    if trunk_controller is not None:
                        step_record[f"{prefix}_trunk_action_diag"] = _action_slice_diag(
                            action, "trunk", trunk_controller
                        )

            def _attached_snapshot(label):
                try:
                    snapshot = self.obtain_attached_object(env, robot)
                    snapshot["attached_obj"] = {
                        str(k): getattr(v, "name", str(v))
                        for k, v in snapshot.get("attached_obj", {}).items()
                    }
                    snapshot["label"] = label
                    return snapshot
                except Exception as e:
                    return {"label": label, "error": f"{type(e).__name__}: {e}"}

            def _toggle_debug_snapshot(obj):
                try:
                    toggle_state = obj.states[object_states.ToggledOn]
                    marker = getattr(toggle_state, "visual_marker", None)
                    marker_pos_np = _debug_to_np(marker.get_position_orientation()[0]) if marker is not None else None
                    marker_quat = marker.get_position_orientation()[1] if marker is not None else None
                    marker_radius = None
                    if marker is not None:
                        try:
                            extent_np = _debug_to_np(getattr(marker, "extent", None))
                            scale_np = _debug_to_np(getattr(marker, "scale", None))
                            if extent_np is not None and scale_np is not None:
                                marker_radius = float(np.min(extent_np * scale_np))
                        except Exception as e:
                            marker_radius = f"ERR:{type(e).__name__}: {e}"
                    marker_debug = {
                        "robot_can_toggle_steps": int(getattr(toggle_state, "robot_can_toggle_steps", -1)),
                        "obj_in_finger_contact_objs": None,
                        "visual_marker_pos": _debug_array_value(marker_pos_np),
                        "visual_marker_quat": _debug_array_value(marker_quat),
                        "visual_marker_radius": marker_radius,
                        "eef_dist_to_marker": {},
                        "finger_min_dist_to_marker": {},
                    }
                    finger_contact_objs = getattr(object_states.ToggledOn, "_finger_contact_objs", None)
                    if finger_contact_objs is not None:
                        try:
                            marker_debug["obj_in_finger_contact_objs"] = obj in finger_contact_objs
                        except Exception as e:
                            marker_debug["obj_in_finger_contact_objs"] = f"ERR:{type(e).__name__}: {e}"
                    for arm_name in ("left", "right"):
                        marker_debug["eef_dist_to_marker"][arm_name] = None
                        marker_debug["finger_min_dist_to_marker"][arm_name] = None
                        if marker_pos_np is None:
                            continue
                        try:
                            eef_pos_np = _debug_to_np(robot.eef_links[arm_name].get_position_orientation()[0])
                            marker_debug["eef_dist_to_marker"][arm_name] = float(np.linalg.norm(eef_pos_np - marker_pos_np))
                        except Exception as e:
                            marker_debug["eef_dist_to_marker"][arm_name] = f"ERR:{type(e).__name__}: {e}"
                        try:
                            finger_records = []
                            for finger_link in getattr(robot, "finger_links", {}).get(arm_name, []):
                                finger_pos_np = _debug_to_np(finger_link.get_position_orientation()[0])
                                finger_records.append(
                                    {
                                        "link": getattr(finger_link, "name", str(finger_link)),
                                        "prim_path": getattr(finger_link, "prim_path", None),
                                        "dist": float(np.linalg.norm(finger_pos_np - marker_pos_np)),
                                    }
                                )
                            if finger_records:
                                marker_debug["finger_min_dist_to_marker"][arm_name] = min(
                                    finger_records, key=lambda item: item["dist"]
                                )
                        except Exception as e:
                            marker_debug["finger_min_dist_to_marker"][arm_name] = f"ERR:{type(e).__name__}: {e}"
                    if overlap_diag_enabled and marker_pos_np is not None and isinstance(marker_radius, float):
                        try:
                            finger_paths = {
                                getattr(finger_link, "prim_path", None)
                                for arm_name, finger_links in getattr(robot, "finger_links", {}).items()
                                if arm_name in overlap_active_arms
                                for finger_link in finger_links
                            }
                            finger_paths.discard(None)
                            overlap_probe_pos_np = marker_pos_np
                            overlap_probe_local_offset_np = None

                            def _probe_overlap_radius(radius):
                                hits = []
                                valid_hit = False

                                def _overlap_report(hit):
                                    nonlocal valid_hit
                                    rigid_body = str(getattr(hit, "rigid_body", ""))
                                    is_finger = rigid_body in finger_paths
                                    valid_hit = valid_hit or is_finger
                                    if len(hits) < overlap_diag_max_hits:
                                        hits.append({"rigid_body": rigid_body, "is_robot_finger": is_finger})
                                    return True

                                og.sim.psqi.overlap_sphere(
                                    radius=radius,
                                    pos=overlap_probe_pos_np.tolist(),
                                    reportFn=_overlap_report,
                                )
                                return {
                                    "radius": radius,
                                    "valid_robot_finger_hit": bool(valid_hit),
                                    "num_recorded_hits": len(hits),
                                    "max_hits": overlap_diag_max_hits,
                                    "hits": hits,
                                }

                            def _classify_rigid_body(rigid_body):
                                if rigid_body in finger_paths:
                                    return "active_finger"
                                if rigid_body in {
                                    getattr(finger_link, "prim_path", None)
                                    for arm_name, finger_links in getattr(robot, "finger_links", {}).items()
                                    if arm_name not in overlap_active_arms
                                    for finger_link in finger_links
                                }:
                                    return "inactive_finger"
                                if str(rigid_body).startswith(str(getattr(obj, "prim_path", ""))):
                                    return "target_obj"
                                if "controllable__" in str(rigid_body):
                                    return "robot_other"
                                return "other"

                            primary_probe = _probe_overlap_radius(marker_radius)
                            extra_radii = [
                                radius for radius in overlap_diag_radii if abs(radius - marker_radius) > 1e-9
                            ]
                            extra_probes = [_probe_overlap_radius(radius) for radius in extra_radii]
                            first_finger_radius = None
                            for probe in [primary_probe] + extra_probes:
                                if probe["valid_robot_finger_hit"]:
                                    first_finger_radius = probe["radius"]
                                    break
                            marker_debug["overlap_sphere_probe"] = {
                                "enabled": True,
                                "radius": marker_radius,
                                "pos": overlap_probe_pos_np.tolist(),
                                "visual_marker_pos": marker_pos_np.tolist(),
                                "local_offset": (
                                    overlap_probe_local_offset_np.tolist()
                                    if overlap_probe_local_offset_np is not None
                                    else None
                                ),
                                "valid_robot_finger_hit": primary_probe["valid_robot_finger_hit"],
                                "num_recorded_hits": primary_probe["num_recorded_hits"],
                                "max_hits": primary_probe["max_hits"],
                                "active_arms": list(overlap_active_arms),
                                "finger_paths": sorted(finger_paths),
                                "hits": primary_probe["hits"],
                                "extra_radius_probes": extra_probes,
                                "first_finger_hit_radius": first_finger_radius,
                            }
                            if neighborhood_diag_enabled and marker_quat is not None:
                                marker_rot_np = np.asarray(T.quat2mat(marker_quat), dtype=float)
                                probe_radius = (
                                    float(neighborhood_diag_radius_raw)
                                    if neighborhood_diag_radius_raw
                                    else marker_radius
                                )
                                directions = [[0.0, 0.0, 0.0]] + neighborhood_diag_dirs
                                samples = []
                                for offset in neighborhood_diag_offsets:
                                    for direction in directions:
                                        direction_np = np.asarray(direction, dtype=float)
                                        direction_norm = float(np.linalg.norm(direction_np))
                                        unit_direction = (
                                            direction_np / direction_norm
                                            if direction_norm > 1e-8
                                            else direction_np
                                        )
                                        sample_pos_np = marker_pos_np + marker_rot_np @ (unit_direction * offset)
                                        sample_hits = []

                                        def _neighborhood_report(hit):
                                            rigid_body = str(getattr(hit, "rigid_body", ""))
                                            if len(sample_hits) < overlap_diag_max_hits:
                                                sample_hits.append(
                                                    {
                                                        "rigid_body": rigid_body,
                                                        "class": _classify_rigid_body(rigid_body),
                                                    }
                                                )
                                            return True

                                        og.sim.psqi.overlap_sphere(
                                            radius=probe_radius,
                                            pos=sample_pos_np.tolist(),
                                            reportFn=_neighborhood_report,
                                        )
                                        hit_classes = sorted({hit["class"] for hit in sample_hits})
                                        samples.append(
                                            {
                                                "offset": float(offset),
                                                "dir_marker_local": unit_direction.tolist(),
                                                "probe_radius": probe_radius,
                                                "pos": sample_pos_np.tolist(),
                                                "hit_classes": hit_classes,
                                                "num_recorded_hits": len(sample_hits),
                                                "hits": sample_hits,
                                            }
                                        )
                                marker_debug["marker_neighborhood_probe"] = {
                                    "enabled": True,
                                    "active_arms": list(overlap_active_arms),
                                    "offsets": neighborhood_diag_offsets,
                                    "probe_radius": probe_radius,
                                    "samples": samples,
                                }
                        except Exception as e:
                            marker_debug["overlap_sphere_probe"] = {
                                "enabled": True,
                                "error": f"{type(e).__name__}: {e}",
                            }
                    return marker_debug
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            def _link_relative_jacobian(control_dict, link):
                link_name = getattr(link, "body_name", None) or getattr(link, "name", None)
                if not link_name:
                    raise ValueError("selected finger link has no name")
                start_idx = 0 if getattr(robot, "fixed_base", False) else 6
                link_idx = robot._articulation_view.get_body_index(link_name)
                return robot.get_relative_jacobian(clone=True)[
                    -(robot.n_links - link_idx),
                    :,
                    start_idx : start_idx + robot.n_joints,
                ]

            def _finger_contact_point_world(finger_link, finger_pos, dtype, device=None):
                offset = finger_jacobian_contact_point_offset.to(dtype=dtype, device=device)
                if float(th.linalg.norm(offset)) <= 0.0:
                    return finger_pos, offset, None
                finger_quat = th.as_tensor(finger_link.get_position_orientation()[1], dtype=dtype, device=device)
                finger_rot = th.as_tensor(T.quat2mat(finger_quat), dtype=dtype, device=device)
                return finger_pos + finger_rot @ offset, offset, finger_rot

            def _gripper_contact_data_diag(arm_name, target_obj, marker_pos_world):
                try:
                    finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                    finger_paths = {getattr(link, "prim_path", None) for link in finger_links}
                    finger_paths.discard(None)
                    target_link_paths = set(getattr(target_obj, "link_prim_paths", []))
                    contacts = []
                    for row, col, force, point, normal, separation in GripperRigidContactAPI.get_contact_data(
                        robot.scene.idx,
                        column_prim_paths=finger_paths,
                    ):
                        row_is_target = row in target_link_paths
                        point_tensor = th.as_tensor(point, dtype=marker_pos_world.dtype, device=marker_pos_world.device)
                        contact = {
                            "row_prim_path": str(row),
                            "col_prim_path": str(col),
                            "row_is_target_obj": bool(row_is_target),
                            "force": _debug_array_value(force),
                            "force_norm": float(th.linalg.norm(th.as_tensor(force, dtype=marker_pos_world.dtype))),
                            "point": _debug_array_value(point),
                            "point_marker_dist": float(th.linalg.norm(point_tensor - marker_pos_world)),
                            "normal": _debug_array_value(normal),
                            "separation": float(th.as_tensor(separation).item()),
                        }
                        contacts.append(contact)
                    target_contacts = [contact for contact in contacts if contact["row_is_target_obj"]]
                    return {
                        "enabled": True,
                        "arm": arm_name,
                        "finger_paths": sorted(finger_paths),
                        "target_link_paths": sorted(target_link_paths),
                        "num_contacts": len(contacts),
                        "num_target_obj_contacts": len(target_contacts),
                        "target_obj_contact": bool(target_contacts),
                        "min_target_contact_point_marker_dist": (
                            None
                            if not target_contacts
                            else min(contact["point_marker_dist"] for contact in target_contacts)
                        ),
                        "contacts": contacts[:8],
                    }
                except Exception as e:
                    return {"enabled": True, "arm": arm_name, "error": f"{type(e).__name__}: {e}"}

            def _frame_projection_diag(delta_world, marker_rot, finger_rot):
                diag = {"delta_world": _debug_array_value(delta_world)}
                if marker_rot is not None:
                    marker_rel = marker_rot.T @ delta_world
                    diag["delta_marker_frame"] = _debug_array_value(marker_rel)
                    diag["marker_axes_world"] = {
                        "x": _debug_array_value(marker_rot[:, 0]),
                        "y": _debug_array_value(marker_rot[:, 1]),
                        "z": _debug_array_value(marker_rot[:, 2]),
                    }
                if finger_rot is not None:
                    finger_rel = finger_rot.T @ delta_world
                    diag["delta_finger_frame"] = _debug_array_value(finger_rel)
                    diag["finger_axes_world"] = {
                        "x": _debug_array_value(finger_rot[:, 0]),
                        "y": _debug_array_value(finger_rot[:, 1]),
                        "z": _debug_array_value(finger_rot[:, 2]),
                    }
                return diag

            def _apply_finger_jacobian_action(
                action, arm_name, finger_link, finger_pos, jacobian_target_pos, control_point_pos, step_record
            ):
                control_dict = robot.get_control_dict()
                arm_controller = robot.controllers.get(f"arm_{arm_name}")
                if arm_controller is None:
                    step_record["finger_jacobian_error"] = "missing_arm_controller"
                    return action
                arm_dof_idx = np.asarray(arm_controller.dof_idx, dtype=int)
                manipulation_dof_idx = arm_dof_idx
                base_controller = robot.controllers.get("base") if finger_jacobian_include_base else None
                base_dof_idx = None
                if base_controller is not None:
                    base_dof_idx = np.asarray(base_controller.dof_idx, dtype=int)
                    manipulation_dof_idx = np.concatenate([base_dof_idx, manipulation_dof_idx])
                trunk_controller = robot.controllers.get("trunk") if finger_jacobian_include_trunk else None
                trunk_dof_idx = None
                if trunk_controller is not None:
                    trunk_dof_idx = np.asarray(trunk_controller.dof_idx, dtype=int)
                    manipulation_dof_idx = np.concatenate([manipulation_dof_idx, trunk_dof_idx])

                jacobian = _link_relative_jacobian(control_dict, finger_link)
                j_pos_origin = np.asarray(_debug_to_np(jacobian[:3, manipulation_dof_idx]), dtype=np.float64)
                j_rot = np.asarray(_debug_to_np(jacobian[3:, manipulation_dof_idx]), dtype=np.float64)

                robot_pos, robot_quat = robot.get_position_orientation()
                finger_rel_pos = T.relative_pose_transform(
                    finger_pos,
                    th.tensor([0.0, 0.0, 0.0, 1.0], dtype=finger_pos.dtype),
                    robot_pos,
                    robot_quat,
                )[0]
                control_point_rel_pos = T.relative_pose_transform(
                    control_point_pos,
                    th.tensor([0.0, 0.0, 0.0, 1.0], dtype=control_point_pos.dtype),
                    robot_pos,
                    robot_quat,
                )[0]
                jacobian_target_rel_pos = T.relative_pose_transform(
                    jacobian_target_pos,
                    th.tensor([0.0, 0.0, 0.0, 1.0], dtype=jacobian_target_pos.dtype),
                    robot_pos,
                    robot_quat,
                )[0]
                contact_offset_rel = control_point_rel_pos - finger_rel_pos
                j_pos = j_pos_origin + np.cross(
                    j_rot.T,
                    np.asarray(_debug_to_np(contact_offset_rel), dtype=np.float64),
                ).T
                axes = finger_jacobian_axes.to(dtype=finger_rel_pos.dtype, device=finger_rel_pos.device)
                bias_world = finger_jacobian_bias.to(dtype=finger_pos.dtype, device=finger_pos.device)
                bias_rel = th.as_tensor(T.quat2mat(robot_quat), dtype=finger_pos.dtype).T @ bias_world
                target_delta = finger_jacobian_gain * (jacobian_target_rel_pos - control_point_rel_pos) * axes + bias_rel
                target_delta_norm = float(th.linalg.norm(target_delta))
                target_delta_clamped = False
                effective_finger_jacobian_step_max_norm = float(
                    (step_record.get("coupled_budget") or {}).get(
                        "finger_jacobian_step_max_norm", finger_jacobian_step_max_norm
                    )
                )
                if (
                    effective_finger_jacobian_step_max_norm > 0.0
                    and target_delta_norm > effective_finger_jacobian_step_max_norm
                ):
                    target_delta = target_delta * (effective_finger_jacobian_step_max_norm / target_delta_norm)
                    target_delta_clamped = True

                jjt = j_pos @ j_pos.T
                damping_eye = (finger_jacobian_damping ** 2) * np.eye(jjt.shape[0])
                try:
                    delta_q = j_pos.T @ np.linalg.solve(
                        jjt + damping_eye,
                        np.asarray(_debug_to_np(target_delta), dtype=np.float64),
                    )
                    solve_status = "ok"
                except np.linalg.LinAlgError:
                    delta_q = np.linalg.pinv(j_pos) @ np.asarray(_debug_to_np(target_delta), dtype=np.float64)
                    solve_status = "pinv_fallback"

                raw_delta_q = np.array(delta_q, copy=True)
                base_dof_count = int(len(base_dof_idx)) if base_dof_idx is not None else 0
                arm_offset = base_dof_count
                trunk_offset = arm_offset + int(len(arm_dof_idx))
                if base_dof_count and finger_jacobian_base_step_max_norm > 0.0:
                    base_delta_norm = float(np.linalg.norm(delta_q[:base_dof_count]))
                    if base_delta_norm > finger_jacobian_base_step_max_norm:
                        delta_q[:base_dof_count] *= finger_jacobian_base_step_max_norm / base_delta_norm
                if finger_jacobian_joint_delta_max_abs > 0.0:
                    delta_q = np.clip(
                        delta_q,
                        -finger_jacobian_joint_delta_max_abs,
                        finger_jacobian_joint_delta_max_abs,
                    )

                q = np.asarray(_debug_to_np(control_dict["joint_position"][manipulation_dof_idx]), dtype=np.float64)
                q_lower = np.asarray(
                    _debug_to_np(
                        arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx]
                    ),
                    dtype=np.float64,
                )
                q_upper = np.asarray(
                    _debug_to_np(
                        arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx]
                    ),
                    dtype=np.float64,
                )
                target_q_unclipped = q + delta_q
                target_q = np.clip(
                    target_q_unclipped,
                    q_lower + finger_jacobian_joint_limit_margin,
                    q_upper - finger_jacobian_joint_limit_margin,
                )

                action_is_tensor = isinstance(action, th.Tensor)
                action_np_dtype = None if action_is_tensor else np.asarray(action).dtype
                joint_position = control_dict.get("joint_position")
                command_dtype = getattr(joint_position, "dtype", th.float32)
                if not isinstance(command_dtype, th.dtype):
                    command_dtype = th.from_numpy(np.empty((), dtype=command_dtype)).dtype
                command_device = getattr(joint_position, "device", None)

                def _command_tensor(values):
                    return th.as_tensor(values, dtype=command_dtype, device=command_device)

                def _assign_action_command(action_idx, command):
                    if action_is_tensor:
                        if not isinstance(command, th.Tensor):
                            command = th.as_tensor(command, dtype=action.dtype, device=action.device)
                        else:
                            command = command.to(dtype=action.dtype, device=action.device)
                        action[action_idx] = command
                    else:
                        action[action_idx] = np.asarray(_debug_to_np(command), dtype=action_np_dtype)

                if base_controller is not None and base_dof_idx is not None:
                    base_command = _command_tensor(delta_q[:base_dof_count])
                    _assign_action_command(
                        robot.controller_action_idx["base"],
                        base_controller._reverse_preprocess_command(base_command),
                    )
                arm_dof_count = int(len(arm_dof_idx))
                _assign_action_command(
                    robot.controller_action_idx[f"arm_{arm_name}"],
                    arm_controller._reverse_preprocess_command(_command_tensor(target_q[arm_offset:trunk_offset])),
                )
                if trunk_controller is not None and trunk_dof_idx is not None:
                    _assign_action_command(
                        robot.controller_action_idx["trunk"],
                        trunk_controller._reverse_preprocess_command(_command_tensor(target_q[trunk_offset:])),
                    )

                step_record["finger_jacobian_enabled"] = True
                step_record["finger_jacobian_solve_status"] = solve_status
                step_record["finger_jacobian_j_shape"] = list(j_pos.shape)
                step_record["finger_jacobian_origin_j_shape"] = list(j_pos_origin.shape)
                step_record["finger_jacobian_rot_j_shape"] = list(j_rot.shape)
                step_record["finger_jacobian_include_base"] = base_controller is not None
                step_record["finger_jacobian_include_trunk"] = trunk_controller is not None
                step_record["finger_jacobian_base_step_max_norm"] = finger_jacobian_base_step_max_norm
                step_record["finger_jacobian_control_point_pos"] = _debug_array_value(control_point_pos)
                step_record["finger_jacobian_contact_point_offset_local"] = _debug_array_value(
                    finger_jacobian_contact_point_offset
                )
                step_record["finger_jacobian_contact_offset_rel"] = _debug_array_value(contact_offset_rel)
                step_record["finger_jacobian_control_point_to_marker_delta"] = _debug_array_value(
                    control_point_pos - jacobian_target_pos
                )
                step_record["finger_jacobian_target_pos"] = _debug_array_value(jacobian_target_pos)
                step_record["finger_jacobian_target_delta_rel"] = _debug_array_value(target_delta)
                step_record["finger_jacobian_target_delta_norm_raw"] = target_delta_norm
                step_record["finger_jacobian_target_delta_clamped"] = target_delta_clamped
                step_record["finger_jacobian_gain"] = finger_jacobian_gain
                step_record["finger_jacobian_axes"] = _debug_array_value(finger_jacobian_axes)
                step_record["finger_jacobian_bias_world"] = _debug_array_value(finger_jacobian_bias)
                step_record["finger_jacobian_damping"] = finger_jacobian_damping
                step_record["finger_jacobian_step_max_norm"] = effective_finger_jacobian_step_max_norm
                step_record["finger_jacobian_config_step_max_norm"] = finger_jacobian_step_max_norm
                step_record["finger_jacobian_joint_delta_max_abs"] = finger_jacobian_joint_delta_max_abs
                step_record["finger_jacobian_delta_q_raw"] = _debug_array_value(raw_delta_q)
                step_record["finger_jacobian_delta_q"] = _debug_array_value(delta_q)
                step_record["finger_jacobian_target_q_unclipped"] = _debug_array_value(target_q_unclipped)
                step_record["finger_jacobian_target_q"] = _debug_array_value(target_q)
                step_record["finger_jacobian_q_clip_delta_norm"] = float(np.linalg.norm(target_q - target_q_unclipped))
                step_record["finger_jacobian_predicted_delta"] = _debug_array_value(j_pos @ delta_q)
                return action

            start_marker_pos = None
            start_ref_obj_pos = None
            prev_marker_pos = None
            best_progress_value = float("inf")
            stale_progress_count = 0
            best_post_step_progress_value = float("inf")
            post_step_regress_count = 0
            post_step_regress_activated = False
            overlap_no_hit_count = 0
            overlap_no_progress_activated = False
            marker_frame_residual_history = []

            if pre_gripper_action is not None and pre_gripper_steps > 0:
                for pre_step in range(pre_gripper_steps):
                    current_left_pos, current_left_quat = robot.get_eef_pose("left")
                    current_right_pos, current_right_quat = robot.get_eef_pose("right")
                    pose = np.zeros((8, 4))
                    pose[:4, :4] = _debug_to_np(T.pose2mat((current_left_pos, current_left_quat)))
                    pose[4:, :4] = _debug_to_np(T.pose2mat((current_right_pos, current_right_quat)))
                    post_mp_action = env_interface.target_pose_to_action(target_pose=pose)
                    post_mp_action = _apply_post_mp_gripper_action(
                        post_mp_action, gripper_action_override=pre_gripper_action
                    )

                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=post_mp_action)
                    post_mp_action = _postprocess_action_compatible(env, post_mp_action)
                    env.step(post_mp_action, video_writer)
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(post_mp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]
                    if pre_step == 0 or (log_interval > 0 and pre_step % log_interval == 0):
                        _record(
                            {
                                "enabled": True,
                                "applied": True,
                                "phase": int(env.execution_phase_ind),
                                "pre_gripper_step": int(pre_step),
                                "active_gripper_action": active_gripper_action,
                                "pre_gripper_action": pre_gripper_action,
                                "pre_gripper_steps": pre_gripper_steps,
                                "toggle_debug": _toggle_debug_snapshot(ref_obj),
                                "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                            }
                        )

            for press_step in range(press_steps):
                try:
                    toggle_state = ref_obj.states[object_states.ToggledOn]
                    marker = getattr(toggle_state, "visual_marker", None)
                    if marker is None:
                        _record(
                            {
                                "enabled": True,
                                "applied": False,
                                "phase": int(env.execution_phase_ind),
                                "press_step": int(press_step),
                                "reason": "missing_visual_marker",
                            }
                        )
                        break
                    marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
                    marker_pos = th.as_tensor(marker_pos_raw, dtype=th.float32)
                    marker_quat = th.as_tensor(marker_quat_raw, dtype=th.float32)
                    marker_rot = th.as_tensor(T.quat2mat(marker_quat), dtype=marker_pos.dtype)
                    if start_marker_pos is None:
                        start_marker_pos = marker_pos.clone()
                    ref_obj_pos_raw, ref_obj_quat_raw = ref_obj.get_position_orientation()
                    ref_obj_pos = th.as_tensor(ref_obj_pos_raw, dtype=marker_pos.dtype)
                    ref_obj_quat = th.as_tensor(ref_obj_quat_raw, dtype=marker_pos.dtype)
                    if start_ref_obj_pos is None:
                        start_ref_obj_pos = ref_obj_pos.clone()
                    marker_drift = float(th.linalg.norm(marker_pos - start_marker_pos))
                    if marker_drift_limit > 0.0 and marker_drift > marker_drift_limit:
                        _record(
                            {
                                "enabled": True,
                                "applied": False,
                                "phase": int(env.execution_phase_ind),
                                "press_step": int(press_step),
                                "reason": "marker_drift_limit_exceeded",
                                "marker_drift_from_start": marker_drift,
                                "marker_drift_limit": marker_drift_limit,
                            }
                        )
                        break

                    current_left_pos, current_left_quat = robot.get_eef_pose("left")
                    current_right_pos, current_right_quat = robot.get_eef_pose("right")
                    pose = np.zeros((8, 4))
                    pose[:4, :4] = _debug_to_np(T.pose2mat((current_left_pos, current_left_quat)))
                    pose[4:, :4] = _debug_to_np(T.pose2mat((current_right_pos, current_right_quat)))

                    step_records = []
                    step_runtime_records = []
                    preservation_records = []
                    for arm_name in active_arms:
                        row_start = 0 if arm_name == "left" else 4
                        eef_pos = th.as_tensor(
                            robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype
                        )
                        finger_links = getattr(robot, "finger_links", {}).get(arm_name, [])
                        if not finger_links:
                            step_records.append({"arm": arm_name, "applied": False, "reason": "no_finger_links"})
                            continue

                        finger_records = []
                        for finger_link in finger_links:
                            finger_pos = th.as_tensor(finger_link.get_position_orientation()[0], dtype=marker_pos.dtype)
                            finger_records.append((finger_link, finger_pos, float(th.linalg.norm(finger_pos - marker_pos))))
                        selected_finger_record = None
                        if force_finger_link:
                            for finger_record in finger_records:
                                finger_name = getattr(finger_record[0], "name", str(finger_record[0]))
                                if force_finger_link == finger_name or force_finger_link in finger_name:
                                    selected_finger_record = finger_record
                                    break
                        finger_link, finger_pos, finger_dist = selected_finger_record or min(
                            finger_records, key=lambda item: item[2]
                        )
                        control_point_pos, contact_offset_local, finger_rot = _finger_contact_point_world(
                            finger_link,
                            finger_pos,
                            marker_pos.dtype,
                            marker_pos.device,
                        )
                        control_point_dist = float(th.linalg.norm(control_point_pos - marker_pos))

                        original_target_pos = np.array(pose[row_start:row_start + 3, 3], copy=True)
                        jacobian_target_pos = marker_pos.clone()
                        if marker_local_offset is not None and marker_rot is not None:
                            jacobian_target_pos = jacobian_target_pos + marker_rot @ marker_local_offset.to(
                                dtype=jacobian_target_pos.dtype, device=jacobian_target_pos.device
                            )
                        if world_offset is not None:
                            jacobian_target_pos = jacobian_target_pos + world_offset.to(
                                dtype=jacobian_target_pos.dtype, device=jacobian_target_pos.device
                            )
                        jacobian_target_pos = jacobian_target_pos + th.tensor(
                            [0.0, 0.0, approach_z], dtype=jacobian_target_pos.dtype
                        )
                        marker_frame_servo_diag = None
                        if marker_frame_servo_enabled and marker_rot is not None:
                            desired_marker_local = th.zeros(3, dtype=marker_pos.dtype, device=marker_pos.device)
                            if marker_local_offset is not None:
                                desired_marker_local = marker_local_offset.to(
                                    dtype=marker_pos.dtype, device=marker_pos.device
                                )
                            desired_marker_local = desired_marker_local + th.tensor(
                                [0.0, 0.0, approach_z], dtype=marker_pos.dtype, device=marker_pos.device
                            )
                            control_point_marker_local = marker_rot.T @ (control_point_pos - marker_pos)
                            marker_frame_residual = control_point_marker_local - desired_marker_local
                            lateral_residual = marker_frame_residual[:2]
                            lateral_axes = marker_frame_servo_lateral_axes.to(
                                dtype=marker_pos.dtype, device=marker_pos.device
                            )
                            weighted_lateral_residual = lateral_residual * lateral_axes
                            lateral_norm = float(th.linalg.norm(weighted_lateral_residual))
                            correction_local = th.zeros(3, dtype=marker_pos.dtype, device=marker_pos.device)
                            y_residual = float(marker_frame_residual[1])
                            residual_norm_for_y_staging = float(th.linalg.norm(marker_frame_residual))
                            y_staging_step_ok = press_step >= marker_frame_servo_y_staging_min_step
                            y_staging_residual_ok = (
                                marker_frame_servo_y_staging_max_residual <= 0.0
                                or residual_norm_for_y_staging <= marker_frame_servo_y_staging_max_residual
                            )
                            y_staging_active = (
                                marker_frame_servo_y_staging_enabled
                                and abs(y_residual) > marker_frame_servo_y_staging_threshold
                                and y_staging_step_ok
                                and y_staging_residual_ok
                            )
                            if y_staging_active:
                                correction_local[1] = -marker_frame_servo_lateral_gain * weighted_lateral_residual[1]
                                normal_active = False
                            else:
                                correction_local[:2] = -marker_frame_servo_lateral_gain * weighted_lateral_residual
                                normal_active = lateral_norm <= marker_frame_servo_lateral_threshold
                                if normal_active:
                                    correction_local[2] = (
                                        -marker_frame_servo_normal_gain * marker_frame_residual[2]
                                    )
                            jacobian_target_pos = control_point_pos + marker_rot @ correction_local
                            if world_offset is not None:
                                jacobian_target_pos = jacobian_target_pos + world_offset.to(
                                    dtype=jacobian_target_pos.dtype, device=jacobian_target_pos.device
                                )
                            marker_frame_servo_diag = {
                                "enabled": True,
                                "desired_marker_local": _debug_array_value(desired_marker_local),
                                "control_point_marker_local": _debug_array_value(control_point_marker_local),
                                "residual_marker_local": _debug_array_value(marker_frame_residual),
                                "weighted_lateral_residual": _debug_array_value(weighted_lateral_residual),
                                "lateral_norm": lateral_norm,
                                "lateral_axes": _debug_array_value(marker_frame_servo_lateral_axes),
                                "lateral_threshold": marker_frame_servo_lateral_threshold,
                                "normal_active": normal_active,
                                "lateral_gain": marker_frame_servo_lateral_gain,
                                "normal_gain": marker_frame_servo_normal_gain,
                                "y_staging_enabled": marker_frame_servo_y_staging_enabled,
                                "y_staging_active": y_staging_active,
                                "y_staging_threshold": marker_frame_servo_y_staging_threshold,
                                "y_staging_min_step": marker_frame_servo_y_staging_min_step,
                                "y_staging_max_residual": marker_frame_servo_y_staging_max_residual,
                                "y_staging_step_ok": y_staging_step_ok,
                                "y_staging_residual_ok": y_staging_residual_ok,
                                "y_residual": y_residual,
                                "residual_norm": residual_norm_for_y_staging,
                                "correction_local": _debug_array_value(correction_local),
                            }
                            if marker_frame_servo_object_drift_comp_enabled and prev_marker_pos is not None:
                                marker_step_delta = marker_pos - prev_marker_pos.to(
                                    dtype=marker_pos.dtype, device=marker_pos.device
                                )
                                jacobian_target_pos = jacobian_target_pos - marker_step_delta
                                marker_frame_servo_diag["object_drift_comp_enabled"] = True
                                marker_frame_servo_diag["object_drift_comp_world"] = _debug_array_value(
                                    -marker_step_delta
                                )
                                marker_frame_servo_diag["marker_step_delta_world"] = _debug_array_value(
                                    marker_step_delta
                                )
                            else:
                                marker_frame_servo_diag["object_drift_comp_enabled"] = (
                                    marker_frame_servo_object_drift_comp_enabled
                                )
                                marker_frame_servo_diag["object_drift_comp_world"] = _debug_array_value(
                                    th.zeros(3, dtype=marker_pos.dtype, device=marker_pos.device)
                                )
                        marker_frame_residual_norm = None
                        marker_frame_residual_stale = False
                        marker_frame_residual_improvement = None
                        control_point_source_residual_norm = None
                        if marker_frame_servo_diag is not None:
                            residual_value = marker_frame_servo_diag.get("residual_marker_local", {}).get("value")
                            if residual_value is not None:
                                marker_frame_residual_norm = float(th.linalg.norm(th.as_tensor(residual_value)))
                                control_point_source_residual_norm = marker_frame_residual_norm
                                if len(marker_frame_residual_history) >= coupled_budget_residual_window:
                                    prior_best = min(marker_frame_residual_history[-coupled_budget_residual_window:])
                                    marker_frame_residual_improvement = prior_best - marker_frame_residual_norm
                                    marker_frame_residual_stale = (
                                        marker_frame_residual_improvement < coupled_budget_residual_eps
                                    )
                        elif marker_local_offset is not None and marker_rot is not None:
                            source_local_residual = (
                                marker_rot.T @ (control_point_pos - marker_pos)
                            ) - marker_local_offset.to(dtype=marker_pos.dtype, device=marker_pos.device)
                            control_point_source_residual_norm = float(th.linalg.norm(source_local_residual))
                        contact_seek_diag = None
                        if contact_seek_enabled and marker_rot is not None and press_step >= contact_seek_min_step:
                            seek_axes = contact_seek_axes.to(dtype=marker_pos.dtype, device=marker_pos.device)
                            marker_to_control_point = control_point_pos - marker_pos
                            control_point_marker_local = marker_rot.T @ marker_to_control_point
                            if contact_seek_dir_local is None:
                                seek_direction_local = control_point_marker_local * seek_axes
                                seek_direction_mode = "control_point_radial"
                            else:
                                seek_direction_local = contact_seek_dir_local.to(
                                    dtype=marker_pos.dtype, device=marker_pos.device
                                ) * seek_axes
                                seek_direction_mode = "fixed_marker_local"
                            seek_direction_norm = float(th.linalg.norm(seek_direction_local))
                            contact_seek_ref_obj_drift = float(th.linalg.norm(ref_obj_pos - start_ref_obj_pos))
                            drift_scale = 1.0
                            drift_stop = (
                                contact_seek_drift_stop_threshold > 0.0
                                and contact_seek_ref_obj_drift >= contact_seek_drift_stop_threshold
                            )
                            if (
                                not drift_stop
                                and contact_seek_drift_scale_threshold > 0.0
                                and contact_seek_ref_obj_drift >= contact_seek_drift_scale_threshold
                            ):
                                if contact_seek_drift_stop_threshold > contact_seek_drift_scale_threshold:
                                    drift_alpha = (
                                        (contact_seek_ref_obj_drift - contact_seek_drift_scale_threshold)
                                        / (contact_seek_drift_stop_threshold - contact_seek_drift_scale_threshold)
                                    )
                                    drift_alpha = min(1.0, max(0.0, drift_alpha))
                                    drift_scale = 1.0 - drift_alpha * (
                                        1.0 - max(0.0, min(1.0, contact_seek_drift_min_scale))
                                    )
                                else:
                                    drift_scale = max(0.0, min(1.0, contact_seek_drift_min_scale))
                            effective_contact_seek_extra_depth = contact_seek_extra_depth * drift_scale
                            if seek_direction_norm > 1e-8:
                                target_radius = (
                                    contact_seek_target_radius
                                    if contact_seek_target_radius > 0.0
                                    else float(th.min(marker.extent * marker.scale).item())
                                )
                                seek_target_local = (
                                    seek_direction_local
                                    / seek_direction_norm
                                    * max(0.0, target_radius - effective_contact_seek_extra_depth)
                                )
                                if not drift_stop:
                                    jacobian_target_pos = marker_pos + marker_rot @ seek_target_local
                                contact_seek_diag = {
                                    "enabled": True,
                                    "active": not drift_stop,
                                    "reason": None if not drift_stop else "ref_obj_drift_stop",
                                    "min_step": contact_seek_min_step,
                                    "target_radius": target_radius,
                                    "extra_depth": contact_seek_extra_depth,
                                    "effective_extra_depth": effective_contact_seek_extra_depth,
                                    "axes": _debug_array_value(contact_seek_axes),
                                    "direction_mode": seek_direction_mode,
                                    "configured_direction_local": _debug_array_value(contact_seek_dir_local),
                                    "ref_obj_drift_from_start": contact_seek_ref_obj_drift,
                                    "drift_scale_threshold": contact_seek_drift_scale_threshold,
                                    "drift_stop_threshold": contact_seek_drift_stop_threshold,
                                    "drift_min_scale": contact_seek_drift_min_scale,
                                    "drift_scale": drift_scale,
                                    "control_point_marker_local": _debug_array_value(control_point_marker_local),
                                    "seek_direction_local": _debug_array_value(seek_direction_local),
                                    "seek_direction_norm": seek_direction_norm,
                                    "seek_target_marker_local": _debug_array_value(seek_target_local),
                                    "jacobian_target_pos": _debug_array_value(jacobian_target_pos),
                                }
                            else:
                                contact_seek_diag = {
                                    "enabled": True,
                                    "active": False,
                                    "reason": "zero_seek_direction",
                                    "min_step": contact_seek_min_step,
                                    "axes": _debug_array_value(contact_seek_axes),
                                    "direction_mode": seek_direction_mode,
                                    "configured_direction_local": _debug_array_value(contact_seek_dir_local),
                                    "control_point_marker_local": _debug_array_value(control_point_marker_local),
                                }
                        elif contact_seek_enabled:
                            contact_seek_diag = {
                                "enabled": True,
                                "active": False,
                                "reason": "before_min_step" if press_step < contact_seek_min_step else "missing_marker_rot",
                                "min_step": contact_seek_min_step,
                            }
                        pre_contact_data_diag = _gripper_contact_data_diag(arm_name, ref_obj, marker_pos)
                        contact_aware_press_diag = {
                            "enabled": contact_aware_press_enabled,
                            "active": False,
                            "reason": None,
                            "min_step": contact_aware_press_min_step,
                            "depth": contact_aware_press_depth,
                            "axes": _debug_array_value(contact_aware_press_axes),
                            "configured_direction_local": _debug_array_value(contact_aware_press_dir_local),
                            "disable_regress_stop": contact_aware_disable_regress_stop,
                        }
                        if contact_aware_press_enabled:
                            target_contact = bool(pre_contact_data_diag.get("target_obj_contact"))
                            precontact_guarded = (
                                contact_aware_allow_precontact
                                and not target_contact
                                and (
                                    contact_aware_precontact_max_dist <= 0.0
                                    or control_point_dist <= contact_aware_precontact_max_dist
                                )
                            )
                            step_ready = press_step >= contact_aware_press_min_step
                            if not step_ready:
                                contact_aware_press_diag["reason"] = "before_min_step"
                            elif not target_contact and not precontact_guarded:
                                contact_aware_press_diag["reason"] = "no_target_contact"
                            elif marker_rot is None:
                                contact_aware_press_diag["reason"] = "missing_marker_rot"
                            else:
                                contact_axes = contact_aware_press_axes.to(
                                    dtype=marker_pos.dtype, device=marker_pos.device
                                )
                                if contact_aware_press_dir_local is not None:
                                    press_direction_local = contact_aware_press_dir_local.to(
                                        dtype=marker_pos.dtype, device=marker_pos.device
                                    ) * contact_axes
                                    direction_mode = "configured"
                                elif contact_seek_dir_local is not None:
                                    press_direction_local = contact_seek_dir_local.to(
                                        dtype=marker_pos.dtype, device=marker_pos.device
                                    ) * contact_axes
                                    direction_mode = "contact_seek"
                                else:
                                    press_direction_local = (control_point_pos - marker_pos)
                                    press_direction_local = marker_rot.T @ press_direction_local
                                    press_direction_local = press_direction_local * contact_axes
                                    direction_mode = "control_point_radial"
                                press_direction_norm = float(th.linalg.norm(press_direction_local))
                                if press_direction_norm > 1e-8 and contact_aware_press_depth > 0.0:
                                    unit_press_direction_local = press_direction_local / press_direction_norm
                                    shell_target_local = None
                                    shell_target_radius = None
                                    if precontact_guarded and contact_aware_precontact_shell_target:
                                        shell_target_radius = (
                                            contact_aware_precontact_shell_radius
                                            if contact_aware_precontact_shell_radius > 0.0
                                            else float(th.min(marker.extent * marker.scale).item())
                                        )
                                        shell_target_radius = max(
                                            0.0,
                                            shell_target_radius - max(0.0, contact_aware_precontact_shell_depth),
                                        )
                                        shell_target_local = unit_press_direction_local * shell_target_radius
                                        press_delta_world = marker_pos + marker_rot @ shell_target_local - control_point_pos
                                        jacobian_target_pos = control_point_pos + press_delta_world
                                        direction_mode = f"{direction_mode}_precontact_shell"
                                    else:
                                        press_delta_world = marker_rot @ (
                                            unit_press_direction_local * contact_aware_press_depth
                                        )
                                        jacobian_target_pos = control_point_pos + press_delta_world
                                    contact_aware_press_diag.update(
                                        {
                                            "active": True,
                                            "reason": (
                                                "precontact_guarded_approach"
                                                if precontact_guarded
                                                else None
                                            ),
                                            "target_contact": target_contact,
                                            "precontact_guarded": precontact_guarded,
                                            "allow_precontact": contact_aware_allow_precontact,
                                            "precontact_max_dist": contact_aware_precontact_max_dist,
                                            "control_point_dist": control_point_dist,
                                            "direction_mode": direction_mode,
                                            "press_direction_local": _debug_array_value(press_direction_local),
                                            "unit_press_direction_local": _debug_array_value(
                                                unit_press_direction_local
                                            ),
                                            "press_direction_norm": press_direction_norm,
                                            "press_delta_world": _debug_array_value(press_delta_world),
                                            "precontact_shell_target": (
                                                precontact_guarded and contact_aware_precontact_shell_target
                                            ),
                                            "precontact_shell_radius": shell_target_radius,
                                            "precontact_shell_depth": contact_aware_precontact_shell_depth,
                                            "precontact_shell_target_local": _debug_array_value(shell_target_local),
                                            "jacobian_target_pos": _debug_array_value(jacobian_target_pos),
                                        }
                                    )
                                else:
                                    contact_aware_press_diag.update(
                                        {
                                            "reason": "zero_direction_or_depth",
                                            "target_contact": target_contact,
                                            "precontact_guarded": precontact_guarded,
                                            "allow_precontact": contact_aware_allow_precontact,
                                            "precontact_max_dist": contact_aware_precontact_max_dist,
                                            "control_point_dist": control_point_dist,
                                            "direction_mode": direction_mode,
                                            "press_direction_local": _debug_array_value(press_direction_local),
                                            "press_direction_norm": press_direction_norm,
                                        }
                                    )
                        corrected_pos = jacobian_target_pos + offset_scale * (eef_pos - finger_pos)
                        residual_comp = th.zeros_like(corrected_pos)
                        residual_comp_clamped = False
                        if residual_comp_gain != 0.0:
                            residual_axes = residual_comp_axes.to(dtype=corrected_pos.dtype, device=corrected_pos.device)
                            residual_comp = -residual_comp_gain * (finger_pos - marker_pos) * residual_axes
                            residual_comp_norm = float(th.linalg.norm(residual_comp))
                            if residual_comp_max_norm > 0.0 and residual_comp_norm > residual_comp_max_norm:
                                residual_comp = residual_comp * (residual_comp_max_norm / residual_comp_norm)
                                residual_comp_clamped = True
                            corrected_pos = corrected_pos + residual_comp
                        micro_servo_delta = th.zeros_like(corrected_pos)
                        if micro_servo_enabled:
                            micro_axes = micro_servo_axes.to(dtype=corrected_pos.dtype, device=corrected_pos.device)
                            micro_bias = micro_servo_bias.to(dtype=corrected_pos.dtype, device=corrected_pos.device)
                            micro_servo_delta = -micro_servo_gain * (finger_pos - marker_pos) * micro_axes + micro_bias
                            corrected_pos = eef_pos + micro_servo_delta
                        unclamped_corrected_pos = corrected_pos.clone()
                        correction_delta_norm = float(th.linalg.norm(corrected_pos - eef_pos))
                        clamped = False
                        effective_max_step = max_step
                        effective_finger_jacobian_step_max_norm = finger_jacobian_step_max_norm
                        coupled_budget_reason = None
                        ref_obj_drift_from_start = float(th.linalg.norm(ref_obj_pos - start_ref_obj_pos))
                        if coupled_budget_enabled:
                            ref_obj_drift_high = (
                                coupled_budget_ref_obj_drift_threshold > 0.0
                                and ref_obj_drift_from_start >= coupled_budget_ref_obj_drift_threshold
                            )
                            if marker_frame_residual_stale or ref_obj_drift_high:
                                scale = min(1.0, max(coupled_budget_min_scale, coupled_budget_scale))
                                effective_max_step = max_step * scale if max_step > 0.0 else max_step
                                effective_finger_jacobian_step_max_norm = (
                                    finger_jacobian_step_max_norm * scale
                                    if finger_jacobian_step_max_norm > 0.0
                                    else finger_jacobian_step_max_norm
                                )
                                reasons = []
                                if marker_frame_residual_stale:
                                    reasons.append("marker_frame_residual_stale")
                                if ref_obj_drift_high:
                                    reasons.append("ref_obj_drift_high")
                                coupled_budget_reason = "+".join(reasons)
                        if (
                            finger_servo_enabled
                            and effective_max_step > 0.0
                            and correction_delta_norm > effective_max_step
                        ):
                            corrected_pos = eef_pos + (corrected_pos - eef_pos) * (effective_max_step / correction_delta_norm)
                            clamped = True
                        pose[row_start:row_start + 3, 3] = _debug_to_np(corrected_pos)
                        desired_source_marker_local = None
                        if marker_rot is not None:
                            desired_source_marker_local = th.zeros(
                                3, dtype=marker_pos.dtype, device=marker_pos.device
                            )
                            if marker_local_offset is not None:
                                desired_source_marker_local = desired_source_marker_local + marker_local_offset.to(
                                    dtype=marker_pos.dtype, device=marker_pos.device
                                )
                            desired_source_marker_local = desired_source_marker_local + th.tensor(
                                [0.0, 0.0, approach_z], dtype=marker_pos.dtype, device=marker_pos.device
                            )
                        all_finger_link_diags = []
                        for record_finger_link, record_finger_pos, record_dist in finger_records:
                            record_control_point_pos, record_contact_offset_local, record_finger_rot = (
                                _finger_contact_point_world(
                                    record_finger_link,
                                    record_finger_pos,
                                    marker_pos.dtype,
                                    marker_pos.device,
                                )
                            )
                            record_diag = {
                                "link": getattr(record_finger_link, "name", str(record_finger_link)),
                                "prim_path": getattr(record_finger_link, "prim_path", None),
                                "dist": record_dist,
                                "selected": record_finger_link is finger_link,
                                "control_point_dist": float(
                                    th.linalg.norm(record_control_point_pos - marker_pos)
                                ),
                                "contact_point_offset_local": _debug_array_value(record_contact_offset_local),
                                "contact_point_offset_world": _debug_array_value(
                                    record_control_point_pos - record_finger_pos
                                ),
                                "finger_pos": _debug_array_value(record_finger_pos),
                                "control_point_pos": _debug_array_value(record_control_point_pos),
                                "control_point_to_marker_projection_diag": _frame_projection_diag(
                                    record_control_point_pos - marker_pos,
                                    marker_rot,
                                    record_finger_rot,
                                ),
                            }
                            if marker_rot is not None:
                                record_control_point_marker_local = marker_rot.T @ (
                                    record_control_point_pos - marker_pos
                                )
                                record_diag["control_point_marker_local"] = _debug_array_value(
                                    record_control_point_marker_local
                                )
                                if desired_source_marker_local is not None:
                                    record_source_residual = (
                                        record_control_point_marker_local - desired_source_marker_local
                                    )
                                    record_diag["source_residual_marker_local"] = _debug_array_value(
                                        record_source_residual
                                    )
                                    record_diag["source_residual_norm"] = float(
                                        th.linalg.norm(record_source_residual)
                                    )
                            all_finger_link_diags.append(record_diag)
                        step_record = {
                            "arm": arm_name,
                            "applied": True,
                            "attached_before_action": _attached_snapshot("before_action"),
                            "finger_servo_enabled": finger_servo_enabled,
                            "finger_link": getattr(finger_link, "name", str(finger_link)),
                            "force_finger_link": force_finger_link or None,
                            "all_finger_link_dists": all_finger_link_diags,
                            "marker_pos": _debug_array_value(marker_pos),
                            "marker_quat": _debug_array_value(marker_quat),
                            "marker_frame_axes_world": {
                                "x": _debug_array_value(marker_rot[:, 0]),
                                "y": _debug_array_value(marker_rot[:, 1]),
                                "z": _debug_array_value(marker_rot[:, 2]),
                            },
                            "marker_drift_from_start": marker_drift,
                            "ref_obj_pos": _debug_array_value(ref_obj_pos),
                            "ref_obj_quat": _debug_array_value(ref_obj_quat),
                            "ref_obj_drift_from_start": float(th.linalg.norm(ref_obj_pos - start_ref_obj_pos)),
                            "finger_pos": _debug_array_value(finger_pos),
                            "finger_prim_path": getattr(finger_link, "prim_path", None),
                            "finger_quat": _debug_array_value(finger_link.get_position_orientation()[1]),
                            "finger_frame_axes_world": (
                                None
                                if finger_rot is None
                                else {
                                    "x": _debug_array_value(finger_rot[:, 0]),
                                    "y": _debug_array_value(finger_rot[:, 1]),
                                    "z": _debug_array_value(finger_rot[:, 2]),
                                }
                            ),
                            "finger_contact_point_offset_local": _debug_array_value(contact_offset_local),
                            "finger_contact_point_offset_world": _debug_array_value(
                                control_point_pos - finger_pos
                            ),
                            "finger_control_point_pos": _debug_array_value(control_point_pos),
                            "eef_pos": _debug_array_value(eef_pos),
                            "finger_to_marker_delta": _debug_array_value(finger_pos - marker_pos),
                            "finger_to_marker_projection_diag": _frame_projection_diag(
                                finger_pos - marker_pos, marker_rot, finger_rot
                            ),
                            "finger_control_point_to_marker_delta": _debug_array_value(
                                control_point_pos - marker_pos
                            ),
                            "jacobian_target_pos": _debug_array_value(jacobian_target_pos),
                            "finger_control_point_to_jacobian_target_delta": _debug_array_value(
                                control_point_pos - jacobian_target_pos
                            ),
                            "finger_control_point_to_marker_projection_diag": _frame_projection_diag(
                                control_point_pos - marker_pos, marker_rot, finger_rot
                            ),
                            "eef_to_marker_delta": _debug_array_value(eef_pos - marker_pos),
                            "finger_marker_dist_before": finger_dist,
                            "finger_control_point_marker_dist_before": control_point_dist,
                            "finger_control_point_source_residual_norm": control_point_source_residual_norm,
                            "original_target_pos": _debug_array_value(original_target_pos),
                            "unclamped_corrected_target_pos": _debug_array_value(unclamped_corrected_pos),
                            "corrected_target_pos": _debug_array_value(corrected_pos),
                            "correction_delta_norm": correction_delta_norm,
                            "max_step": effective_max_step,
                            "config_max_step": max_step,
                            "clamped": clamped,
                            "coupled_budget": {
                                "enabled": coupled_budget_enabled,
                                "reason": coupled_budget_reason,
                                "residual_window": coupled_budget_residual_window,
                                "residual_eps": coupled_budget_residual_eps,
                                "scale": coupled_budget_scale,
                                "min_scale": coupled_budget_min_scale,
                                "ref_obj_drift_threshold": coupled_budget_ref_obj_drift_threshold,
                                "marker_frame_residual_norm": marker_frame_residual_norm,
                                "marker_frame_residual_stale": marker_frame_residual_stale,
                                "marker_frame_residual_improvement": marker_frame_residual_improvement,
                                "ref_obj_drift_from_start": ref_obj_drift_from_start,
                                "finger_servo_max_step": effective_max_step,
                                "finger_jacobian_step_max_norm": effective_finger_jacobian_step_max_norm,
                            },
                            "contact_gate": {
                                "enabled": contact_gate_enabled,
                                "min_step": contact_gate_min_step,
                                "residual_threshold": contact_gate_residual_threshold,
                                "ref_obj_drift_threshold": contact_gate_ref_obj_drift_threshold,
                                "marker_z_lift_threshold": contact_gate_marker_z_lift_threshold,
                                "ref_obj_z_lift_threshold": contact_gate_ref_obj_z_lift_threshold,
                                "require_residual_stale": contact_gate_require_residual_stale,
                                "source_residual_norm": control_point_source_residual_norm,
                                "marker_z_lift_from_start": (
                                    None
                                    if start_marker_pos is None
                                    else float(marker_pos[2] - start_marker_pos[2])
                                ),
                                "ref_obj_z_lift_from_start": float(ref_obj_pos[2] - start_ref_obj_pos[2]),
                                "ref_obj_drift_from_start": ref_obj_drift_from_start,
                                "marker_frame_residual_stale": marker_frame_residual_stale,
                            },
                            "offset_scale": offset_scale,
                            "approach_z": approach_z,
                            "marker_local_offset": _debug_array_value(marker_local_offset),
                            "world_offset": _debug_array_value(world_offset),
                            "residual_comp_gain": residual_comp_gain,
                            "residual_comp_axes": _debug_array_value(residual_comp_axes),
                            "residual_comp_max_norm": residual_comp_max_norm,
                            "residual_comp": _debug_array_value(residual_comp),
                            "residual_comp_clamped": residual_comp_clamped,
                            "micro_servo_enabled": micro_servo_enabled,
                            "micro_servo_gain": micro_servo_gain,
                            "micro_servo_axes": _debug_array_value(micro_servo_axes),
                            "micro_servo_bias": _debug_array_value(micro_servo_bias),
                            "micro_servo_delta": _debug_array_value(micro_servo_delta),
                            "contact_seek": contact_seek_diag,
                            "pre_step_gripper_contact_data": pre_contact_data_diag,
                            "contact_aware_press": contact_aware_press_diag,
                            "marker_frame_servo": marker_frame_servo_diag,
                            "hold_left_gripper_action": hold_left_gripper_action,
                            "hold_right_gripper_action": hold_right_gripper_action,
                        }
                        step_records.append(step_record)
                        step_runtime_records.append((step_record, arm_name, finger_link, jacobian_target_pos))

                    if preserve_ref_obj_pose_enabled:
                        ref_obj_drift = ref_obj_pos - start_ref_obj_pos
                        axes = preserve_ref_obj_pose_axes.to(dtype=ref_obj_drift.dtype, device=ref_obj_drift.device)
                        correction = -preserve_ref_obj_pose_gain * ref_obj_drift * axes
                        correction_norm_raw = float(th.linalg.norm(correction))
                        correction_clamped = False
                        if preserve_ref_obj_pose_max_step > 0.0 and correction_norm_raw > preserve_ref_obj_pose_max_step:
                            correction = correction * (preserve_ref_obj_pose_max_step / correction_norm_raw)
                            correction_clamped = True
                        for preserve_arm_name in preserve_ref_obj_pose_arms:
                            row_start = 0 if preserve_arm_name == "left" else 4
                            if preserve_arm_name == "left":
                                arm_pos = th.as_tensor(current_left_pos, dtype=marker_pos.dtype)
                            else:
                                arm_pos = th.as_tensor(current_right_pos, dtype=marker_pos.dtype)
                            target_pos = arm_pos + correction.to(dtype=arm_pos.dtype, device=arm_pos.device)
                            pose[row_start:row_start + 3, 3] = _debug_to_np(target_pos)
                            preservation_records.append(
                                {
                                    "enabled": True,
                                    "arm": preserve_arm_name,
                                    "ref_obj_pos": _debug_array_value(ref_obj_pos),
                                    "ref_obj_quat": _debug_array_value(ref_obj_quat),
                                    "start_ref_obj_pos": _debug_array_value(start_ref_obj_pos),
                                    "ref_obj_drift_world": _debug_array_value(ref_obj_drift),
                                    "ref_obj_drift_norm": float(th.linalg.norm(ref_obj_drift)),
                                    "gain": preserve_ref_obj_pose_gain,
                                    "axes": _debug_array_value(preserve_ref_obj_pose_axes),
                                    "max_step": preserve_ref_obj_pose_max_step,
                                    "correction_world": _debug_array_value(correction),
                                    "correction_norm_raw": correction_norm_raw,
                                    "correction_clamped": correction_clamped,
                                    "arm_pos": _debug_array_value(arm_pos),
                                    "target_pos": _debug_array_value(target_pos),
                                }
                            )
                        for step_record in step_records:
                            step_record["ref_obj_pose_preservation"] = list(preservation_records)

                    if log_only:
                        post_mp_action = None
                        for step_record in step_records:
                            if step_record.get("applied"):
                                step_record["log_only"] = True
                    else:
                        post_mp_action = env_interface.target_pose_to_action(target_pose=pose)
                    if post_mp_action is not None and finger_jacobian_enabled:
                        for step_record, arm_name, finger_link, jacobian_target_pos in step_runtime_records:
                            if not step_record.get("applied"):
                                continue
                            try:
                                finger_pos_for_jacobian = th.as_tensor(
                                    finger_link.get_position_orientation()[0], dtype=marker_pos.dtype
                                )
                                control_point_pos_for_jacobian = _finger_contact_point_world(
                                    finger_link,
                                    finger_pos_for_jacobian,
                                    marker_pos.dtype,
                                    marker_pos.device,
                                )[0]
                                post_mp_action = _apply_finger_jacobian_action(
                                    post_mp_action,
                                    arm_name,
                                    finger_link,
                                    finger_pos_for_jacobian,
                                    jacobian_target_pos,
                                    control_point_pos_for_jacobian,
                                    step_record,
                                )
                            except Exception as e:
                                step_record["finger_jacobian_enabled"] = True
                                step_record["finger_jacobian_error"] = f"{type(e).__name__}: {e}"
                    if post_mp_action is not None:
                        post_mp_action = _apply_post_mp_gripper_action(post_mp_action)
                        for step_record, arm_name, _, _ in step_runtime_records:
                            _record_post_mp_action_diag(post_mp_action, step_record, arm_name, "pre_postprocess")
                            action_np = np.asarray(_debug_to_np(post_mp_action), dtype=float)
                            step_record["pre_postprocess_gripper_action"] = {
                                "left": float(action_np[env_interface.gripper_action_dim[0]]),
                                "right": float(action_np[env_interface.gripper_action_dim[1]]),
                            }

                        post_mp_action = _postprocess_action_compatible(env, post_mp_action)
                        for step_record, arm_name, _, _ in step_runtime_records:
                            _record_post_mp_action_diag(post_mp_action, step_record, arm_name, "post_postprocess")
                            action_np = np.asarray(_debug_to_np(post_mp_action), dtype=float)
                            step_record["post_postprocess_gripper_action"] = {
                                "left": float(action_np[env_interface.gripper_action_dim[0]]),
                                "right": float(action_np[env_interface.gripper_action_dim[1]]),
                            }
                            step_record["action_repeat_steps"] = action_repeat_steps

                    state = None
                    obs = None
                    obs_info = None
                    datagen_info = None
                    if post_mp_action is not None:
                        for repeat_step in range(action_repeat_steps):
                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=post_mp_action)
                            env.step(post_mp_action, video_writer)
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(post_mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            if repeat_step != action_repeat_steps - 1:
                                cur_success_metrics = env.is_success()
                                for k in success:
                                    success[k] = success[k] or cur_success_metrics[k]
                    try:
                        post_marker_pos_raw = marker.get_position_orientation()[0]
                        post_marker_quat_raw = marker.get_position_orientation()[1]
                        post_ref_obj_pos_raw, post_ref_obj_quat_raw = ref_obj.get_position_orientation()
                        post_marker_pos = th.as_tensor(post_marker_pos_raw, dtype=marker_pos.dtype)
                        post_marker_quat = th.as_tensor(post_marker_quat_raw, dtype=marker_pos.dtype)
                        post_ref_obj_pos = th.as_tensor(post_ref_obj_pos_raw, dtype=marker_pos.dtype)
                        post_ref_obj_quat = th.as_tensor(post_ref_obj_quat_raw, dtype=marker_pos.dtype)
                        post_marker_rot = th.as_tensor(T.quat2mat(post_marker_quat), dtype=marker_pos.dtype)
                        for step_record, arm_name, finger_link, _ in step_runtime_records:
                            step_record["attached_after_action"] = _attached_snapshot("after_action")
                            post_finger_pos = th.as_tensor(
                                finger_link.get_position_orientation()[0], dtype=marker_pos.dtype
                            )
                            post_control_point_pos, _, post_finger_rot = _finger_contact_point_world(
                                finger_link,
                                post_finger_pos,
                                marker_pos.dtype,
                                marker_pos.device,
                            )
                            post_eef_pos = th.as_tensor(
                                robot.eef_links[arm_name].get_position_orientation()[0], dtype=marker_pos.dtype
                            )
                            step_record["post_step_marker_pos"] = _debug_array_value(post_marker_pos)
                            step_record["post_step_marker_quat"] = _debug_array_value(post_marker_quat)
                            step_record["post_step_marker_frame_axes_world"] = {
                                "x": _debug_array_value(post_marker_rot[:, 0]),
                                "y": _debug_array_value(post_marker_rot[:, 1]),
                                "z": _debug_array_value(post_marker_rot[:, 2]),
                            }
                            step_record["post_step_ref_obj_pos"] = _debug_array_value(post_ref_obj_pos)
                            step_record["post_step_ref_obj_quat"] = _debug_array_value(post_ref_obj_quat)
                            step_record["post_step_ref_obj_delta_world"] = _debug_array_value(
                                post_ref_obj_pos - ref_obj_pos
                            )
                            step_record["post_step_ref_obj_drift_from_start"] = float(
                                th.linalg.norm(post_ref_obj_pos - start_ref_obj_pos)
                            )
                            step_record["post_step_finger_pos"] = _debug_array_value(post_finger_pos)
                            step_record["post_step_finger_quat"] = _debug_array_value(
                                finger_link.get_position_orientation()[1]
                            )
                            step_record["post_step_finger_frame_axes_world"] = (
                                None
                                if post_finger_rot is None
                                else {
                                    "x": _debug_array_value(post_finger_rot[:, 0]),
                                    "y": _debug_array_value(post_finger_rot[:, 1]),
                                    "z": _debug_array_value(post_finger_rot[:, 2]),
                                }
                            )
                            step_record["post_step_finger_control_point_pos"] = _debug_array_value(
                                post_control_point_pos
                            )
                            step_record["post_step_finger_contact_point_offset_world"] = _debug_array_value(
                                post_control_point_pos - post_finger_pos
                            )
                            step_record["post_step_eef_pos"] = _debug_array_value(post_eef_pos)
                            step_record["post_step_finger_to_marker_delta"] = _debug_array_value(
                                post_finger_pos - post_marker_pos
                            )
                            step_record["post_step_finger_to_marker_projection_diag"] = _frame_projection_diag(
                                post_finger_pos - post_marker_pos, post_marker_rot, post_finger_rot
                            )
                            step_record["post_step_finger_control_point_to_marker_delta"] = _debug_array_value(
                                post_control_point_pos - post_marker_pos
                            )
                            step_record[
                                "post_step_finger_control_point_to_marker_projection_diag"
                            ] = _frame_projection_diag(
                                post_control_point_pos - post_marker_pos,
                                post_marker_rot,
                                post_finger_rot,
                            )
                            step_record["post_step_eef_to_marker_delta"] = _debug_array_value(
                                post_eef_pos - post_marker_pos
                            )
                            step_record["post_step_finger_marker_dist"] = float(
                                th.linalg.norm(post_finger_pos - post_marker_pos)
                            )
                            step_record["post_step_finger_control_point_marker_dist"] = float(
                                th.linalg.norm(post_control_point_pos - post_marker_pos)
                            )
                            pre_finger_pos = th.as_tensor(
                                step_record["finger_pos"]["value"], dtype=post_finger_pos.dtype
                            )
                            pre_control_point_pos = th.as_tensor(
                                step_record["finger_control_point_pos"]["value"],
                                dtype=post_control_point_pos.dtype,
                            )
                            actual_delta_world = post_finger_pos - pre_finger_pos
                            actual_control_point_delta_world = post_control_point_pos - pre_control_point_pos
                            marker_delta_world = post_marker_pos - marker_pos
                            robot_pos, robot_quat = robot.get_position_orientation()
                            actual_delta_rel = th.as_tensor(
                                T.quat2mat(robot_quat), dtype=actual_delta_world.dtype
                            ).T @ actual_delta_world
                            actual_control_point_delta_rel = th.as_tensor(
                                T.quat2mat(robot_quat), dtype=actual_control_point_delta_world.dtype
                            ).T @ actual_control_point_delta_world
                            marker_delta_rel = th.as_tensor(
                                T.quat2mat(robot_quat), dtype=marker_delta_world.dtype
                            ).T @ marker_delta_world
                            step_record["post_step_finger_delta_world"] = _debug_array_value(actual_delta_world)
                            step_record["post_step_finger_control_point_delta_world"] = _debug_array_value(
                                actual_control_point_delta_world
                            )
                            step_record["post_step_marker_delta_world"] = _debug_array_value(marker_delta_world)
                            step_record["post_step_finger_delta_rel"] = _debug_array_value(actual_delta_rel)
                            step_record["post_step_finger_control_point_delta_rel"] = _debug_array_value(
                                actual_control_point_delta_rel
                            )
                            step_record["post_step_marker_delta_rel"] = _debug_array_value(marker_delta_rel)
                            if "finger_jacobian_predicted_delta" in step_record:
                                predicted_delta = th.as_tensor(
                                    step_record["finger_jacobian_predicted_delta"]["value"],
                                    dtype=actual_delta_rel.dtype,
                                )
                                actual_minus_predicted = actual_delta_rel - predicted_delta
                                actual_control_point_minus_predicted = actual_control_point_delta_rel - predicted_delta
                                step_record["finger_jacobian_actual_minus_predicted_delta"] = _debug_array_value(
                                    actual_minus_predicted
                                )
                                step_record[
                                    "finger_jacobian_control_point_actual_minus_predicted_delta"
                                ] = _debug_array_value(actual_control_point_minus_predicted)
                                step_record["finger_jacobian_actual_delta_norm"] = float(
                                    th.linalg.norm(actual_delta_rel)
                                )
                                step_record["finger_jacobian_control_point_actual_delta_norm"] = float(
                                    th.linalg.norm(actual_control_point_delta_rel)
                                )
                                step_record["finger_jacobian_predicted_delta_norm"] = float(
                                    th.linalg.norm(predicted_delta)
                                )
                                step_record["finger_jacobian_actual_predicted_error_norm"] = float(
                                    th.linalg.norm(actual_minus_predicted)
                                )
                                step_record[
                                    "finger_jacobian_control_point_actual_predicted_error_norm"
                                ] = float(th.linalg.norm(actual_control_point_minus_predicted))
                            if isinstance(step_record.get("finger_marker_dist_before"), (int, float)):
                                step_record["post_step_finger_marker_dist_improvement"] = float(
                                    step_record["finger_marker_dist_before"]
                                    - step_record["post_step_finger_marker_dist"]
                                )
                            if isinstance(
                                step_record.get("finger_control_point_marker_dist_before"),
                                (int, float),
                            ):
                                step_record["post_step_finger_control_point_marker_dist_improvement"] = float(
                                    step_record["finger_control_point_marker_dist_before"]
                                    - step_record["post_step_finger_control_point_marker_dist"]
                                )
                            step_record["post_step_gripper_contact_data"] = _gripper_contact_data_diag(
                                arm_name,
                                ref_obj,
                                post_marker_pos,
                            )
                    except Exception as e:
                        for step_record, _, _, _ in step_runtime_records:
                            step_record["post_step_measure_error"] = f"{type(e).__name__}: {e}"
                    if enable_marker_vis:
                        env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                        env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                        left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3], dtype=th.float32)))
                        right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3], dtype=th.float32)))
                        env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                        env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)

                    cur_success_metrics = env.is_success()
                    if not log_only:
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                    post_toggle_debug = _toggle_debug_snapshot(ref_obj)
                    robot_can_toggle_steps = post_toggle_debug.get("robot_can_toggle_steps")
                    if predicate_hold_enabled and isinstance(robot_can_toggle_steps, int):
                        predicate_task_success = bool(cur_success_metrics.get("task", False))
                        if predicate_task_success or robot_can_toggle_steps >= predicate_hold_target_steps:
                            _record(
                                {
                                    "enabled": True,
                                    "applied": True,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": (
                                        "task_success_after_predicate_hold"
                                        if predicate_task_success
                                        else "predicate_hold_target_reached"
                                    ),
                                    "predicate_hold_enabled": True,
                                    "robot_can_toggle_steps": robot_can_toggle_steps,
                                    "predicate_hold_target_steps": predicate_hold_target_steps,
                                    "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                                    "toggle_debug": post_toggle_debug,
                                }
                            )
                            break
                        if robot_can_toggle_steps > 0:
                            _record(
                                {
                                    "enabled": True,
                                    "applied": True,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": "predicate_hold_continue",
                                    "predicate_hold_enabled": True,
                                    "robot_can_toggle_steps": robot_can_toggle_steps,
                                    "predicate_hold_target_steps": predicate_hold_target_steps,
                                    "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                                    "toggle_debug": post_toggle_debug,
                                }
                            )
                            prev_marker_pos = marker_pos.clone()
                            continue

                    should_log = press_step == 0 or (log_interval > 0 and press_step % log_interval == 0)
                    if should_log:
                        _record(
                            {
                                "enabled": True,
                                "applied": True,
                                "phase": int(env.execution_phase_ind),
                                "press_step": int(press_step),
                                "records": step_records,
                                "preservation_records": preservation_records,
                                "active_gripper_action": active_gripper_action,
                                "log_only": log_only,
                                "toggle_debug": post_toggle_debug,
                                "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                            }
                        )
                    active_progress_values = []
                    live_marker_radius = None
                    overlap_has_active_hit = None
                    contact_gate_should_abort = False
                    contact_gate_records = []
                    if contact_gate_enabled and press_step >= contact_gate_min_step:
                        for record in step_records:
                            if not record.get("applied"):
                                continue
                            gate = record.get("contact_gate") or {}
                            source_residual = gate.get("source_residual_norm")
                            ref_obj_drift_value = gate.get("ref_obj_drift_from_start")
                            marker_z_lift = gate.get("marker_z_lift_from_start")
                            ref_obj_z_lift = gate.get("ref_obj_z_lift_from_start")
                            residual_missed = (
                                isinstance(source_residual, (int, float))
                                and contact_gate_residual_threshold > 0.0
                                and source_residual > contact_gate_residual_threshold
                            )
                            drift_high = (
                                isinstance(ref_obj_drift_value, (int, float))
                                and contact_gate_ref_obj_drift_threshold > 0.0
                                and ref_obj_drift_value >= contact_gate_ref_obj_drift_threshold
                            )
                            marker_lift_high = (
                                isinstance(marker_z_lift, (int, float))
                                and contact_gate_marker_z_lift_threshold > 0.0
                                and marker_z_lift >= contact_gate_marker_z_lift_threshold
                            )
                            ref_obj_lift_high = (
                                isinstance(ref_obj_z_lift, (int, float))
                                and contact_gate_ref_obj_z_lift_threshold > 0.0
                                and ref_obj_z_lift >= contact_gate_ref_obj_z_lift_threshold
                            )
                            residual_stale_ok = (
                                not contact_gate_require_residual_stale
                                or bool(gate.get("marker_frame_residual_stale"))
                            )
                            gate_triggered = bool(
                                residual_missed
                                and residual_stale_ok
                                and (drift_high or marker_lift_high or ref_obj_lift_high)
                            )
                            gate.update(
                                {
                                    "residual_missed": residual_missed,
                                    "drift_high": drift_high,
                                    "marker_lift_high": marker_lift_high,
                                    "ref_obj_lift_high": ref_obj_lift_high,
                                    "residual_stale_ok": residual_stale_ok,
                                    "triggered": gate_triggered,
                                }
                            )
                            record["contact_gate"] = gate
                            contact_gate_records.append(
                                {
                                    "arm": record.get("arm"),
                                    "source_residual_norm": source_residual,
                                    "ref_obj_drift_from_start": ref_obj_drift_value,
                                    "marker_z_lift_from_start": marker_z_lift,
                                    "ref_obj_z_lift_from_start": ref_obj_z_lift,
                                    "residual_missed": residual_missed,
                                    "drift_high": drift_high,
                                    "marker_lift_high": marker_lift_high,
                                    "ref_obj_lift_high": ref_obj_lift_high,
                                    "residual_stale_ok": residual_stale_ok,
                                    "triggered": gate_triggered,
                                }
                            )
                            contact_gate_should_abort = contact_gate_should_abort or gate_triggered
                    if contact_gate_should_abort:
                        _record(
                            {
                                "enabled": True,
                                "applied": False,
                                "phase": int(env.execution_phase_ind),
                                "press_step": int(press_step),
                                "reason": "contact_geometry_gate_abort",
                                "contact_gate_records": contact_gate_records,
                                "success": {k: bool(v) for k, v in cur_success_metrics.items()},
                                "toggle_debug": post_toggle_debug,
                            }
                        )
                        break
                    if progress_metric == "overlap_radius":
                        overlap_probe = post_toggle_debug.get("overlap_sphere_probe") or {}
                        probe_radius = overlap_probe.get("radius")
                        debug_marker_radius = post_toggle_debug.get("visual_marker_radius")
                        if isinstance(probe_radius, (int, float)):
                            live_marker_radius = float(probe_radius)
                        elif isinstance(debug_marker_radius, (int, float)):
                            live_marker_radius = float(debug_marker_radius)
                        first_hit_radius = overlap_probe.get("first_finger_hit_radius")
                        if isinstance(first_hit_radius, (int, float)):
                            overlap_has_active_hit = True
                            overlap_no_hit_count = 0
                            overlap_no_progress_activated = True
                            active_progress_values.append(float(first_hit_radius))
                        elif isinstance(live_marker_radius, float):
                            overlap_has_active_hit = False
                            if press_step >= no_progress_min_step and (
                                not no_progress_activate_on_first_hit or overlap_no_progress_activated
                            ):
                                overlap_no_hit_count += 1
                            active_progress_values.append(float("inf"))
                    else:
                        active_progress_values = [
                            record["finger_marker_dist_before"]
                            for record in step_records
                            if record.get("applied")
                            and isinstance(record.get("finger_marker_dist_before"), (int, float))
                        ]
                    if active_progress_values:
                        current_best = min(active_progress_values)
                        post_step_progress_values = [
                            record["post_step_finger_marker_dist"]
                            for record in step_records
                            if record.get("applied")
                            and isinstance(record.get("post_step_finger_marker_dist"), (int, float))
                        ]
                        post_step_current_best = (
                            min(post_step_progress_values)
                            if post_step_progress_values
                            else None
                        )
                        no_progress_tracking_active = press_step >= no_progress_min_step and (
                            progress_metric != "overlap_radius"
                            or not no_progress_activate_on_first_hit
                            or overlap_no_progress_activated
                        )
                        if no_progress_tracking_active:
                            if current_best < best_progress_value - no_progress_epsilon:
                                best_progress_value = current_best
                                stale_progress_count = 0
                            else:
                                stale_progress_count += 1
                        progress_converged = (
                            current_best <= convergence_threshold
                            if progress_metric == "finger_dist"
                            else isinstance(live_marker_radius, float)
                            and current_best <= float(live_marker_radius) + max(0.0, convergence_threshold)
                        )
                        post_step_regress_tracking_active = (
                            post_step_regress_enabled
                            and post_step_regress_steps > 0
                            and post_step_current_best is not None
                            and press_step >= post_step_regress_min_step
                            and not (
                                contact_aware_disable_regress_stop
                                and any(
                                    bool((record.get("contact_aware_press") or {}).get("active"))
                                    for record in step_records
                                    if record.get("applied")
                                )
                            )
                        )
                        if post_step_regress_tracking_active:
                            if post_step_current_best < best_post_step_progress_value - post_step_regress_epsilon:
                                best_post_step_progress_value = post_step_current_best
                                post_step_regress_count = 0
                            else:
                                post_step_regress_count += 1
                            if (
                                post_step_regress_activation_threshold <= 0.0
                                or best_post_step_progress_value <= post_step_regress_activation_threshold
                            ):
                                post_step_regress_activated = True
                        if convergence_threshold > 0.0 and progress_converged:
                            _record(
                                {
                                    "enabled": True,
                                    "applied": True,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": (
                                        "finger_converged"
                                        if progress_metric == "finger_dist"
                                        else "overlap_radius_converged"
                                    ),
                                    "progress_metric": progress_metric,
                                    "current_progress_value": current_best,
                                    "best_progress_value": best_progress_value,
                                    "convergence_threshold": convergence_threshold,
                                    "marker_radius": live_marker_radius,
                                }
                            )
                            break
                        if (
                            post_step_regress_tracking_active
                            and post_step_regress_activated
                            and post_step_regress_count >= post_step_regress_steps
                        ):
                            _record(
                                {
                                    "enabled": True,
                                    "applied": False,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": "post_step_finger_regress_stop",
                                    "progress_metric": progress_metric,
                                    "current_progress_value": current_best,
                                    "post_step_current_progress_value": post_step_current_best,
                                    "best_progress_value": best_progress_value,
                                    "best_post_step_progress_value": best_post_step_progress_value,
                                    "post_step_regress_count": post_step_regress_count,
                                    "post_step_regress_steps": post_step_regress_steps,
                                    "post_step_regress_epsilon": post_step_regress_epsilon,
                                    "post_step_regress_min_step": post_step_regress_min_step,
                                    "post_step_regress_activation_threshold": (
                                        post_step_regress_activation_threshold
                                    ),
                                    "marker_radius": live_marker_radius,
                                }
                            )
                            break
                        if (
                            progress_metric == "overlap_radius"
                            and no_progress_steps > 0
                            and overlap_has_active_hit is False
                            and overlap_no_hit_count >= no_progress_steps
                        ):
                            _record(
                                {
                                    "enabled": True,
                                    "applied": False,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": "overlap_radius_no_hit",
                                    "progress_metric": progress_metric,
                                    "current_progress_value": current_best,
                                    "best_progress_value": best_progress_value,
                                    "overlap_no_hit_count": overlap_no_hit_count,
                                    "no_progress_min_step": no_progress_min_step,
                                    "no_progress_after_first_hit": no_progress_activate_on_first_hit,
                                    "no_progress_steps": no_progress_steps,
                                    "no_progress_epsilon": no_progress_epsilon,
                                    "marker_radius": live_marker_radius,
                                }
                            )
                            break
                        if no_progress_steps > 0 and stale_progress_count >= no_progress_steps:
                            _record(
                                {
                                    "enabled": True,
                                    "applied": False,
                                    "phase": int(env.execution_phase_ind),
                                    "press_step": int(press_step),
                                    "reason": (
                                        "finger_no_progress"
                                        if progress_metric == "finger_dist"
                                        else "overlap_radius_no_progress"
                                    ),
                                    "progress_metric": progress_metric,
                                    "current_progress_value": current_best,
                                    "best_progress_value": best_progress_value,
                                    "stale_progress_count": stale_progress_count,
                                    "no_progress_min_step": no_progress_min_step,
                                    "no_progress_after_first_hit": no_progress_activate_on_first_hit,
                                    "no_progress_steps": no_progress_steps,
                                    "no_progress_epsilon": no_progress_epsilon,
                                    "marker_radius": live_marker_radius,
                                }
                            )
                            break
                    for record in step_records:
                        coupled_budget = record.get("coupled_budget") or {}
                        residual_norm = coupled_budget.get("marker_frame_residual_norm")
                        if isinstance(residual_norm, (int, float)) and math.isfinite(residual_norm):
                            marker_frame_residual_history.append(float(residual_norm))
                    prev_marker_pos = marker_pos.clone()
                except Exception as e:
                    _record(
                        {
                            "enabled": True,
                            "applied": False,
                            "phase": int(env.execution_phase_ind),
                            "press_step": int(press_step),
                            "reason": f"post_mp_press_error:{type(e).__name__}: {e}",
                        }
                    )
                    break
            return local_env_step

        def _debug_array_value(x):
            arr = _debug_to_np(x)
            if arr is None:
                return None
            return {
                "shape": list(arr.shape),
                "value": arr.tolist(),
                "finite": bool(np.isfinite(arr).all()),
            }

        def _debug_pose_value(pose):
            if pose is None:
                return None
            if isinstance(pose, dict):
                return {str(k): _debug_pose_value(v) for k, v in pose.items()}
            try:
                pos, quat = pose
                return {"pos": _debug_array_value(pos), "quat": _debug_array_value(quat)}
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}", "raw": str(pose)}

        def _log_nav_debug(stage, **kwargs):
            record = {
                "phase": int(env.execution_phase_ind),
                "stage": stage,
                "global_env_step": int(env.global_env_step),
            }
            record.update(kwargs)
            phase_logs[env.execution_phase_ind].setdefault("nav_debug", []).append(record)
            print("[MOMAGEN_NAV_DEBUG] " + json.dumps(record, default=str), flush=True)
            return record

        def _csv_env_matches(value, csv_value):
            if csv_value is None or str(csv_value).strip() == "":
                return True
            tokens = [token.strip() for token in str(csv_value).split(",") if token.strip()]
            return str(value) in tokens

        def _nav_arm_mp_feasibility_filter_enabled():
            enabled = bool(
                int(
                    os.environ.get(
                        "MOMAGEN_NAV_REQUIRE_ARM_MP_FEASIBILITY",
                        os.environ.get("MOMAGEN_NAV_FILTER_BY_UPCOMING_ARM_MP", "0"),
                    )
                    or 0
                )
            )
            if not enabled:
                return False

            task_filter = os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_TASK_FILTER") or os.environ.get(
                "MOMAGEN_NAV_ARM_MP_FEASIBILITY_TASKS"
            )
            if task_filter:
                env_name = getattr(env, "name", None) or getattr(env, "_name", None) or ""
                if str(task_filter) not in str(env_name):
                    return False

            phase_filter = os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_PHASES")
            if phase_filter and not _csv_env_matches(int(env.execution_phase_ind), phase_filter):
                return False

            object_filter = os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_OBJECTS")
            if object_filter:
                object_names = [name for name in (object_ref or {}).values() if name is not None]
                if not any(str(obj_filter).strip() in object_names for obj_filter in object_filter.split(",")):
                    return False

            return True

        def _build_nav_arm_mp_probe_targets(eef_pose_arg):
            target_pos = {}
            target_quat = {}

            def _add_arm_target(arm_name, pose):
                if pose is None:
                    return
                link_name = robot.eef_link_names[arm_name]
                pos, quat = pose
                target_pos[link_name] = th.as_tensor(pos, dtype=th.float32)
                target_quat[link_name] = th.as_tensor(quat, dtype=th.float32)

            # Match the downstream manipulation branch. If ARM_NO_TORSO exists,
            # inactive arms are removed from the target set. On older OG configs
            # without ARM_NO_TORSO, downstream uses ARM and keeps inactive arms at
            # their current EEF poses so CuRobo's main ee_link target remains valid.
            if object_ref.get("arm_left") is not None:
                pose = eef_pose_arg.get("left") if isinstance(eef_pose_arg, dict) else eef_pose_arg
                _add_arm_target("left", pose)
            if object_ref.get("arm_right") is not None:
                pose = eef_pose_arg.get("right") if isinstance(eef_pose_arg, dict) else eef_pose_arg
                _add_arm_target("right", pose)
            if not hasattr(CuRoboEmbodimentSelection, "ARM_NO_TORSO"):
                if object_ref.get("arm_left") is None:
                    _add_arm_target("left", robot.get_eef_pose("left"))
                if object_ref.get("arm_right") is None:
                    _add_arm_target("right", robot.get_eef_pose("right"))

            return target_pos, target_quat

        def _candidate_curobo_base_pose(candidate_pose, emb_sel):
            """Approximate the CuRobo base link pose after moving the holonomic base to candidate_pose."""
            mg = getattr(env, "cmg", None)
            curobo_base_link_name = getattr(mg, "base_link", {}).get(emb_sel) if mg is not None else None
            if curobo_base_link_name is None or curobo_base_link_name not in robot.links:
                return None, None

            current_joint_pos = robot.get_joint_positions()
            base_control_idx = th.as_tensor(robot.base_control_idx).long()
            current_base_q = current_joint_pos[base_control_idx].detach().cpu().float()
            candidate_base_q = th.as_tensor(candidate_pose, dtype=th.float32).detach().cpu().float()

            current_link_pos, current_link_quat = robot.links[curobo_base_link_name].get_position_orientation()
            current_link_pos = th.as_tensor(current_link_pos, dtype=th.float32).detach().cpu()
            current_link_quat = th.as_tensor(current_link_quat, dtype=th.float32).detach().cpu()

            def _base_frame_from_xy_yaw(base_q):
                pos = th.tensor([base_q[0], base_q[1], current_link_pos[2]], dtype=th.float32)
                quat = T.euler2quat(th.tensor([0.0, 0.0, base_q[2]], dtype=th.float32))
                return T.pose2mat((pos, quat))

            current_base_pose = _base_frame_from_xy_yaw(current_base_q)
            candidate_base_pose = _base_frame_from_xy_yaw(candidate_base_q)
            current_link_pose = T.pose2mat((current_link_pos, current_link_quat))
            # Preserve any fixed offset between the canonical holonomic base frame and CuRobo's actual base link.
            base_to_curobo_link = T.pose_inv(current_base_pose) @ current_link_pose
            candidate_link_pose = candidate_base_pose @ base_to_curobo_link
            return candidate_link_pose[:3, 3], T.mat2quat(candidate_link_pose[:3, :3])

        def _targets_in_candidate_curobo_base_frame(target_pos_single, target_quat_single, candidate_pose, emb_sel, batch_size):
            candidate_base_pos, candidate_base_quat = _candidate_curobo_base_pose(candidate_pose, emb_sel)
            if candidate_base_pos is None or candidate_base_quat is None:
                raise RuntimeError(f"cannot infer candidate CuRobo base pose for emb_sel={emb_sel}")
            inv_candidate_base_pose = T.pose_inv(T.pose2mat((candidate_base_pos, candidate_base_quat)))

            target_pos_local = {}
            target_quat_local = {}
            for link_name, world_pos in target_pos_single.items():
                world_quat = target_quat_single[link_name]
                world_pose = T.pose2mat(
                    (
                        th.as_tensor(world_pos, dtype=th.float32).detach().cpu(),
                        th.as_tensor(world_quat, dtype=th.float32).detach().cpu(),
                    )
                )
                local_pose = inv_candidate_base_pose @ world_pose
                local_pos = local_pose[:3, 3]
                local_quat = T.mat2quat(local_pose[:3, :3])
                target_pos_local[link_name] = th.stack([local_pos for _ in range(batch_size)])
                target_quat_local[link_name] = th.stack([local_quat for _ in range(batch_size)])

            return target_pos_local, target_quat_local

        def _candidate_robot_link_pose_after_base_move(candidate_pose, link_name, emb_sel):
            """Approximate a robot link pose after moving only the holonomic base to candidate_pose."""
            mg = getattr(env, "cmg", None)
            curobo_base_link_name = getattr(mg, "base_link", {}).get(emb_sel) if mg is not None else None
            if curobo_base_link_name is None or curobo_base_link_name not in robot.links or link_name not in robot.links:
                return None, None

            candidate_base_pos, candidate_base_quat = _candidate_curobo_base_pose(candidate_pose, emb_sel)
            if candidate_base_pos is None or candidate_base_quat is None:
                return None, None

            current_base_pos, current_base_quat = robot.links[curobo_base_link_name].get_position_orientation()
            current_link_pos, current_link_quat = robot.links[link_name].get_position_orientation()
            current_base_pose = T.pose2mat(
                (
                    th.as_tensor(current_base_pos, dtype=th.float32).detach().cpu(),
                    th.as_tensor(current_base_quat, dtype=th.float32).detach().cpu(),
                )
            )
            current_link_pose = T.pose2mat(
                (
                    th.as_tensor(current_link_pos, dtype=th.float32).detach().cpu(),
                    th.as_tensor(current_link_quat, dtype=th.float32).detach().cpu(),
                )
            )
            candidate_base_pose = T.pose2mat((candidate_base_pos, candidate_base_quat))
            base_to_link = T.pose_inv(current_base_pose) @ current_link_pose
            candidate_link_pose = candidate_base_pose @ base_to_link
            return candidate_link_pose[:3, 3], T.mat2quat(candidate_link_pose[:3, :3])

        def _candidate_active_eef_reach_score(candidate_pose, target_pos_single, target_quat_single, emb_sel):
            """Score how close active EEFs would be to their targets after a base-only move."""
            active_scores = []
            active_details = {}
            orientation_weight = float(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_ORI_WEIGHT", "0.0") or 0.0)

            for arm_name in ("left", "right"):
                if object_ref.get(f"arm_{arm_name}") is None:
                    continue
                link_name = robot.eef_link_names.get(arm_name)
                if link_name is None or link_name not in target_pos_single:
                    continue
                candidate_link_pos, candidate_link_quat = _candidate_robot_link_pose_after_base_move(
                    candidate_pose,
                    link_name,
                    emb_sel,
                )
                if candidate_link_pos is None or candidate_link_quat is None:
                    active_details[link_name] = {"error": "candidate_link_pose_unavailable"}
                    continue

                target_pos = th.as_tensor(target_pos_single[link_name], dtype=th.float32).detach().cpu()
                candidate_link_pos = th.as_tensor(candidate_link_pos, dtype=th.float32).detach().cpu()
                pos_dist = float(th.norm(candidate_link_pos - target_pos).item())
                score = pos_dist
                quat_angle = None
                if orientation_weight > 0.0 and link_name in target_quat_single:
                    try:
                        target_quat = th.as_tensor(target_quat_single[link_name], dtype=th.float32).detach().cpu()
                        candidate_link_quat = th.as_tensor(candidate_link_quat, dtype=th.float32).detach().cpu()
                        dot = float(th.abs(th.dot(candidate_link_quat, target_quat)).clamp(max=1.0).item())
                        quat_angle = float(2.0 * math.acos(dot))
                        score += orientation_weight * quat_angle
                    except Exception as e:
                        active_details[link_name] = {"pos_dist": pos_dist, "quat_error": f"{type(e).__name__}: {e}"}
                        active_scores.append(score)
                        continue

                active_scores.append(score)
                active_details[link_name] = {
                    "pos_dist": pos_dist,
                    "quat_angle": quat_angle,
                    "score": score,
                    "candidate_pos": _debug_array_value(candidate_link_pos),
                    "target_pos": _debug_array_value(target_pos),
                }

            if not active_scores:
                return None, active_details

            aggregate = str(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_SCORE_AGG", "max") or "max").lower()
            if aggregate == "mean":
                score = float(np.mean(active_scores))
            elif aggregate == "sum":
                score = float(np.sum(active_scores))
            else:
                score = float(np.max(active_scores))
                aggregate = "max"
            return {"value": score, "aggregate": aggregate, "orientation_weight": orientation_weight}, active_details

        def _eef_aligned_base_pose(eef_pose_arg, selected_nav_arm, emb_sel):
            """Return a base pose that would place the current selected EEF at the target pose without arm motion."""
            if selected_nav_arm not in ("left", "right"):
                return None
            mg = getattr(env, "cmg", None)
            curobo_base_link_name = getattr(mg, "base_link", {}).get(emb_sel) if mg is not None else None
            eef_link_name = robot.eef_link_names.get(selected_nav_arm)
            if curobo_base_link_name is None or curobo_base_link_name not in robot.links or eef_link_name not in robot.links:
                return None
            target_pose = eef_pose_arg.get(selected_nav_arm) if isinstance(eef_pose_arg, dict) else eef_pose_arg
            if target_pose is None:
                return None

            current_base_pos, current_base_quat = robot.links[curobo_base_link_name].get_position_orientation()
            current_eef_pos, current_eef_quat = robot.links[eef_link_name].get_position_orientation()
            current_base_pose = T.pose2mat(
                (
                    th.as_tensor(current_base_pos, dtype=th.float32).detach().cpu(),
                    th.as_tensor(current_base_quat, dtype=th.float32).detach().cpu(),
                )
            )
            current_eef_pose = T.pose2mat(
                (
                    th.as_tensor(current_eef_pos, dtype=th.float32).detach().cpu(),
                    th.as_tensor(current_eef_quat, dtype=th.float32).detach().cpu(),
                )
            )
            target_eef_pose = T.pose2mat(
                (
                    th.as_tensor(target_pose[0], dtype=th.float32).detach().cpu(),
                    th.as_tensor(target_pose[1], dtype=th.float32).detach().cpu(),
                )
            )
            base_to_eef = T.pose_inv(current_base_pose) @ current_eef_pose
            desired_base_pose = target_eef_pose @ T.pose_inv(base_to_eef)
            desired_yaw = T.quat2euler(T.mat2quat(desired_base_pose[:3, :3]))[2]
            return th.tensor([desired_base_pose[0, 3], desired_base_pose[1, 3], desired_yaw], dtype=th.float32)

        def _apply_nav_arm_mp_feasibility_filter(
            validate_result,
            candidate_poses,
            eef_pose_arg,
            plan_with_open_gripper=False,
            skip_obstacle_update=False,
            compute_trajectories_fn=None,
        ):
            """Reject navigation base candidates that cannot run the upcoming exact ARM MP.

            This is an env-gated MoMaGen-local consistency check. It does not execute
            any actions and does not replace base navigation; it only tightens the pose
            acceptance predicate used by OG sampling so the accepted base pose is also
            feasible for the downstream ARM_NO_TORSO + attached-object planner.
            """
            if not _nav_arm_mp_feasibility_filter_enabled():
                return validate_result
            if compute_trajectories_fn is None:
                compute_trajectories_fn = getattr(env.cmg, "compute_trajectories", None)
            if compute_trajectories_fn is None:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_skipped",
                    reason="compute_trajectories_unavailable",
                    base_mp_trial=int(base_mp_trial),
                )
                return validate_result

            result = validate_result.clone() if hasattr(validate_result, "clone") else copy.copy(validate_result)
            result_np = _debug_to_np(result)
            if result_np is None:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_skipped",
                    reason="invalid_validate_result",
                    base_mp_trial=int(base_mp_trial),
                )
                return validate_result

            target_pos_single, target_quat_single = _build_nav_arm_mp_probe_targets(eef_pose_arg)
            if not target_pos_single:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_skipped",
                    reason="empty_probe_targets",
                    base_mp_trial=int(base_mp_trial),
                )
                return validate_result

            try:
                retval = self.obtain_attached_object(env, robot)
                probe_attached_obj = retval["attached_obj"]
                probe_attached_obj_scale = retval["attached_obj_scale"]
            except Exception as e:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_skipped",
                    reason="obtain_attached_object_failed",
                    error=f"{type(e).__name__}: {e}",
                    base_mp_trial=int(base_mp_trial),
                )
                return validate_result

            ignore_attached_obj_probe = bool(int(os.environ.get("MOMAGEN_IGNORE_ATTACHED_OBJ_FOR_ARM_MP", "0") or 0))
            if ignore_attached_obj_probe and probe_attached_obj:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_ignore_attached_obj",
                    base_mp_trial=int(base_mp_trial),
                    attached_obj={str(k): getattr(v, "name", str(v)) for k, v in probe_attached_obj.items()},
                )
                probe_attached_obj = None
                probe_attached_obj_scale = None

            attached_filter = os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_ATTACHED")
            if attached_filter:
                attached_names = [getattr(obj, "name", str(obj)) for obj in (probe_attached_obj or {}).values()]
                if not any(
                    token.strip() and any(token.strip() in attached_name for attached_name in attached_names)
                    for token in attached_filter.split(",")
                ):
                    _log_nav_debug(
                        "nav_arm_mp_feasibility_filter_skipped",
                        reason="attached_filter_mismatch",
                        attached_names=attached_names,
                        attached_filter=attached_filter,
                        base_mp_trial=int(base_mp_trial),
                    )
                    return validate_result

            batch_size = int(getattr(env.primitive._motion_generator, "batch_size", 1) or 1)
            arm_no_torso_emb_sel = getattr(
                CuRoboEmbodimentSelection,
                "ARM_NO_TORSO",
                CuRoboEmbodimentSelection.ARM,
            )
            if arm_no_torso_emb_sel not in getattr(env.cmg, "mg", {}):
                arm_no_torso_emb_sel = CuRoboEmbodimentSelection.ARM
            fallback_emb_sels = []
            if (
                not hasattr(CuRoboEmbodimentSelection, "ARM_NO_TORSO")
                and arm_no_torso_emb_sel != CuRoboEmbodimentSelection.DEFAULT
                and CuRoboEmbodimentSelection.DEFAULT in getattr(env.cmg, "mg", {})
            ):
                fallback_emb_sels.append(CuRoboEmbodimentSelection.DEFAULT)
            if plan_with_open_gripper and hasattr(env.primitive, "_get_joint_position_with_fingers_at_limit"):
                current_joint_pos = env.primitive._get_joint_position_with_fingers_at_limit("upper")
            else:
                current_joint_pos = robot.get_joint_positions()
            base_control_idx = th.as_tensor(robot.base_control_idx).long()
            target_frame = str(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_TARGET_FRAME", "world") or "world").lower()
            if target_frame not in {"world", "candidate_local"}:
                _log_nav_debug(
                    "nav_arm_mp_feasibility_filter_bad_target_frame",
                    requested_target_frame=target_frame,
                    fallback_target_frame="world",
                    base_mp_trial=int(base_mp_trial),
                )
                target_frame = "world"
            max_attempts = int(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_MAX_ATTEMPTS", "50") or 50)
            timeout = float(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_TIMEOUT", "60.0") or 60.0)
            ik_fail_return = int(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_IK_FAIL_RETURN", "10") or 10)
            enable_finetune = bool(int(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_FINETUNE", "1") or 1))
            ik_only_probe = bool(int(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_IK_ONLY", "0") or 0))
            probe_records = []

            for idx, candidate_pose in enumerate(candidate_poses):
                og_valid = bool(np.asarray(result_np).astype(bool)[idx])
                record = {
                    "candidate_index": int(idx),
                    "candidate_pose": _debug_array_value(candidate_pose),
                    "og_valid": og_valid,
                    "arm_mp_feasible": None,
                    "status": None,
                }
                if not og_valid:
                    probe_records.append(record)
                    continue

                try:
                    candidate_joint_pos = current_joint_pos.clone()
                    candidate_pose_t = th.as_tensor(candidate_pose, dtype=candidate_joint_pos.dtype, device=candidate_joint_pos.device)
                    candidate_joint_pos[base_control_idx] = candidate_pose_t
                    reach_score, reach_details = _candidate_active_eef_reach_score(
                        candidate_pose,
                        target_pos_single,
                        target_quat_single,
                        arm_no_torso_emb_sel,
                    )
                    record.update(
                        {
                            "active_eef_reach_score": reach_score,
                            "active_eef_reach_details": reach_details,
                        }
                    )
                    feasible = False
                    status_value = None
                    successes_np = None
                    attempted_emb_sels = [arm_no_torso_emb_sel]
                    attempted_statuses = []
                    for probe_emb_sel in attempted_emb_sels:
                        probe_target_pos_single = copy.copy(target_pos_single)
                        probe_target_quat_single = copy.copy(target_quat_single)
                        if not hasattr(CuRoboEmbodimentSelection, "ARM_NO_TORSO"):
                            # Downstream manipulation keeps inactive-arm targets at the EEF pose *after* navigation
                            # when ARM_NO_TORSO is unavailable. During this dry-run, the simulator has not moved yet,
                            # so approximate those inactive targets from the candidate base pose instead of pinning the
                            # inactive arm to its pre-navigation world pose (which makes otherwise trivial constraints
                            # look unreachable).
                            for inactive_arm in ("left", "right"):
                                if object_ref.get(f"arm_{inactive_arm}") is not None:
                                    continue
                                inactive_link_name = robot.eef_link_names.get(inactive_arm)
                                if inactive_link_name is None:
                                    continue
                                candidate_link_pos, candidate_link_quat = _candidate_robot_link_pose_after_base_move(
                                    candidate_pose,
                                    inactive_link_name,
                                    probe_emb_sel,
                                )
                                if candidate_link_pos is not None and candidate_link_quat is not None:
                                    probe_target_pos_single[inactive_link_name] = candidate_link_pos
                                    probe_target_quat_single[inactive_link_name] = candidate_link_quat
                        if target_frame == "candidate_local":
                            target_pos, target_quat = _targets_in_candidate_curobo_base_frame(
                                probe_target_pos_single,
                                probe_target_quat_single,
                                candidate_pose=candidate_pose,
                                emb_sel=probe_emb_sel,
                                batch_size=batch_size,
                            )
                            is_local_probe = True
                        else:
                            # Match OG's own _validate_poses / _target_in_reach_of_robot dry-run semantics:
                            # keep the target in the world frame and pass candidate_joint_pos so CuRobo updates
                            # locked base joints for the hypothetical base pose. Using a hand-rolled candidate-local
                            # transform can over-constrain the root frame and incorrectly turn OG-valid poses into
                            # IK_FAIL before the real navigation pose has been applied in sim.
                            target_pos = {
                                link_name: th.stack(
                                    [th.as_tensor(world_pos, dtype=th.float32) for _ in range(batch_size)]
                                )
                                for link_name, world_pos in probe_target_pos_single.items()
                            }
                            target_quat = {
                                link_name: th.stack(
                                    [th.as_tensor(world_quat, dtype=th.float32) for _ in range(batch_size)]
                                )
                                for link_name, world_quat in probe_target_quat_single.items()
                            }
                            is_local_probe = False
                        attached_obj_options = None
                        if probe_attached_obj:
                            attached_obj_options = {}
                            for link_name in probe_attached_obj.keys():
                                candidate_link_pos, candidate_link_quat = _candidate_robot_link_pose_after_base_move(
                                    candidate_pose,
                                    link_name,
                                    probe_emb_sel,
                                )
                                if candidate_link_pos is not None and candidate_link_quat is not None:
                                    attached_obj_options[link_name] = {
                                        "ee_pose": (candidate_link_pos, candidate_link_quat),
                                    }
                            attached_obj_options = _attached_payload_options(
                                probe_attached_obj,
                                base_options=attached_obj_options,
                            )
                        mp_probe = compute_trajectories_fn(
                            target_pos=target_pos,
                            target_quat=target_quat,
                            initial_joint_pos=candidate_joint_pos,
                            is_local=is_local_probe,
                            max_attempts=max_attempts,
                            timeout=timeout,
                            ik_fail_return=ik_fail_return,
                            enable_finetune_trajopt=enable_finetune,
                            finetune_attempts=1 if enable_finetune else 0,
                            return_full_result=True,
                            success_ratio=1.0 / batch_size,
                            attached_obj=probe_attached_obj,
                            attached_obj_scale=probe_attached_obj_scale,
                            attached_obj_options=attached_obj_options,
                            skip_obstacle_update=skip_obstacle_update,
                            ik_only=ik_only_probe,
                            emb_sel=probe_emb_sel,
                        )
                        if isinstance(mp_probe, tuple):
                            successes = mp_probe[0]
                            current_status_value = None
                        else:
                            successes = getattr(mp_probe[0], "success", None) if len(mp_probe) > 0 else None
                            status_obj = getattr(mp_probe[0], "status", None) if len(mp_probe) > 0 else None
                            current_status_value = getattr(status_obj, "value", str(status_obj))
                        successes_np = _debug_to_np(successes)
                        feasible = bool(successes_np is not None and np.asarray(successes_np).astype(bool).any())
                        status_value = f"{probe_emb_sel}:{current_status_value}"
                        attempted_statuses.append(
                            {
                                "emb_sel": str(probe_emb_sel),
                                "status": current_status_value,
                                "successes": None if successes_np is None else np.asarray(successes_np).astype(bool).tolist(),
                            }
                        )
                        if feasible:
                            break
                        if (
                            fallback_emb_sels
                            and probe_emb_sel == arm_no_torso_emb_sel
                            and current_status_value is not None
                            and "IK Fail" in str(current_status_value)
                        ):
                            attempted_emb_sels.extend(fallback_emb_sels)
                    record.update(
                        {
                            "arm_mp_feasible": feasible,
                            "status": status_value,
                            "successes": None if successes_np is None else np.asarray(successes_np).astype(bool).tolist(),
                            "attempted_emb_sels": [str(emb_sel) for emb_sel in attempted_emb_sels],
                            "attempted_statuses": attempted_statuses,
                            "target_frame": target_frame,
                            "ik_only": ik_only_probe,
                            "ignored_attached_obj_for_arm_mp": ignore_attached_obj_probe,
                        }
                    )
                    if not feasible:
                        result[idx] = False
                except TypeError as e:
                    if "unexpected keyword" in str(e) or "got an unexpected" in str(e):
                        _log_nav_debug(
                            "nav_arm_mp_feasibility_filter_skipped",
                            reason="compute_trajectories_api_mismatch",
                            error=f"{type(e).__name__}: {e}",
                            base_mp_trial=int(base_mp_trial),
                        )
                        return validate_result
                    record.update({"arm_mp_feasible": False, "status": f"ERR:{type(e).__name__}: {e}"})
                    result[idx] = False
                except Exception as e:
                    record.update({"arm_mp_feasible": False, "status": f"ERR:{type(e).__name__}: {e}"})
                    result[idx] = False
                probe_records.append(record)

            pick_best_by_reach = bool(
                int(os.environ.get("MOMAGEN_NAV_ARM_MP_FEASIBILITY_PICK_BEST_REACH", "1") or 1)
            )
            best_candidate_index = None
            if pick_best_by_reach:
                filtered_np_before_best = _debug_to_np(result)
                if filtered_np_before_best is not None:
                    valid_indices = [
                        int(i)
                        for i, valid in enumerate(np.asarray(filtered_np_before_best).astype(bool).reshape(-1))
                        if bool(valid)
                    ]
                    if valid_indices:
                        score_by_index = {}
                        for record in probe_records:
                            idx = record.get("candidate_index")
                            score_record = record.get("active_eef_reach_score")
                            score_value = None
                            if isinstance(score_record, dict):
                                score_value = score_record.get("value")
                            if score_value is None:
                                score_value = float("inf")
                            score_by_index[int(idx)] = float(score_value)
                        best_candidate_index = min(valid_indices, key=lambda i: (score_by_index.get(i, float("inf")), i))
                        if len(valid_indices) > 1:
                            for candidate_index in valid_indices:
                                if candidate_index != best_candidate_index:
                                    result[candidate_index] = False
                        for record in probe_records:
                            idx = int(record.get("candidate_index"))
                            record["selected_by_best_reach_score"] = idx == best_candidate_index

            filtered_np = _debug_to_np(result)
            _log_nav_debug(
                "nav_arm_mp_feasibility_filter_result",
                base_mp_trial=int(base_mp_trial),
                emb_sel=str(arm_no_torso_emb_sel),
                target_links=list(target_pos_single.keys()),
                main_ee_link=getattr(env.cmg, "ee_link", {}).get(arm_no_torso_emb_sel),
                curobo_base_link=getattr(env.cmg, "base_link", {}).get(arm_no_torso_emb_sel),
                attached_obj=(
                    None
                    if not probe_attached_obj
                    else {str(k): getattr(v, "name", str(v)) for k, v in probe_attached_obj.items()}
                ),
                og_valid_count=int(np.asarray(result_np).astype(bool).sum()),
                filtered_valid_count=(None if filtered_np is None else int(np.asarray(filtered_np).astype(bool).sum())),
                pick_best_by_reach=pick_best_by_reach,
                best_candidate_index=best_candidate_index,
                records=probe_records,
            )
            return result

        def _get_nav_debug_robot_state(selected_nav_arm=None, eef_pose_arg=None):
            primitive_arm = getattr(env.primitive, "arm", None)
            try:
                obj_in_hand = env.primitive._get_obj_in_hand()
            except Exception as e:
                obj_in_hand = f"ERR:{type(e).__name__}: {e}"

            actual_grasping_arms = []
            obj_in_hand_name = getattr(obj_in_hand, "name", str(obj_in_hand)) if obj_in_hand is not None else None

            grasp_state = {}
            for arm_name in ("left", "right"):
                candidate_obj = None
                obj_name = object_ref.get(f"arm_{arm_name}") if object_ref is not None else None
                if obj_name is not None:
                    try:
                        candidate_obj = env.env.scene.object_registry("name", obj_name)
                    except Exception:
                        candidate_obj = None
                try:
                    grasp = str(robot.is_grasping(arm=arm_name))
                except Exception as e:
                    grasp = f"ERR:{type(e).__name__}: {e}"
                try:
                    candidate_grasp = (
                        None
                        if candidate_obj is None
                        else str(robot.is_grasping(arm=arm_name, candidate_obj=candidate_obj))
                    )
                except Exception as e:
                    candidate_grasp = f"ERR:{type(e).__name__}: {e}"
                grasp_state[arm_name] = {
                    "object_ref": obj_name,
                    "is_grasping": grasp,
                    "is_grasping_ref_obj": candidate_grasp,
                    "eef_link_name": robot.eef_link_names.get(arm_name) if hasattr(robot, "eef_link_names") else None,
                }
                if "TRUE" in str(grasp):
                    actual_grasping_arms.append(arm_name)

            try:
                robot_base_pos = robot.get_position_orientation()[0]
                robot_base_room = robot.scene._seg_map.get_room_instance_by_point(robot_base_pos[:2])
            except Exception as e:
                robot_base_room = f"ERR:{type(e).__name__}: {e}"

            try:
                robot_joint_positions = _debug_to_np(robot.get_joint_positions())
                base_control_idx = _debug_to_np(robot.base_control_idx)
                base_idx = _debug_to_np(getattr(robot, "base_idx", None))
                base_control_joint_pos = (
                    None
                    if robot_joint_positions is None or base_control_idx is None
                    else robot_joint_positions[base_control_idx.astype(int)].tolist()
                )
                base_idx_joint_pos = (
                    None
                    if robot_joint_positions is None or base_idx is None
                    else robot_joint_positions[base_idx.astype(int)].tolist()
                )
            except Exception as e:
                base_control_joint_pos = f"ERR:{type(e).__name__}: {e}"
                base_idx_joint_pos = f"ERR:{type(e).__name__}: {e}"

            attached_link_if_og_self_arm = None
            if primitive_arm is not None and hasattr(robot, "eef_link_names"):
                try:
                    attached_link_if_og_self_arm = robot.eef_link_names[primitive_arm]
                except Exception as e:
                    attached_link_if_og_self_arm = f"ERR:{type(e).__name__}: {e}"

            return {
                "selected_nav_arm": selected_nav_arm,
                "primitive_arm": primitive_arm,
                "primitive_arm_eef_link": attached_link_if_og_self_arm,
                "object_ref": copy.deepcopy(object_ref),
                "ref_obj_name": getattr(ref_obj, "name", None),
                "ref_obj_pose": _debug_pose_record(ref_obj.get_position_orientation, "nav.ref_obj_pose", []),
                "robot_base_pose": _debug_pose_record(robot.get_position_orientation, "nav.robot_base_pose", []),
                "robot_base_room": robot_base_room,
                "robot_base_control_joint_pos": base_control_joint_pos,
                "robot_base_idx_joint_pos": base_idx_joint_pos,
                "eef_pose_arg": _debug_pose_value(eef_pose_arg),
                "obj_in_hand": obj_in_hand_name,
                "obj_in_hand_root_link": (
                    getattr(getattr(obj_in_hand, "root_link", None), "name", None)
                    if not isinstance(obj_in_hand, str)
                    else None
                ),
                "actual_grasping_arms": actual_grasping_arms,
                "grasp_state": grasp_state,
            }

        def _log_traversability_debug(stage, target_pose_2d=None, candidate_poses=None):
            """Log traversability-map evidence for the current base and sampled / planned base targets."""
            if not bool(int(os.environ.get("MOMAGEN_DEBUG_TRAV", "0") or 0)):
                return None

            def _floor_for_z(z, trav_map):
                floor_heights = getattr(trav_map, "floor_heights", None) or [0.0]
                try:
                    heights = [float(h) for h in floor_heights]
                    return int(np.argmin(np.abs(np.asarray(heights, dtype=float) - float(z))))
                except Exception:
                    return 0

            def _nearest_traversable(trav_map, eroded_map, xy_map):
                try:
                    traversable_pixels = th.where(eroded_map != 0)
                    if len(traversable_pixels) < 2 or traversable_pixels[0].numel() == 0:
                        return None
                    pixels = th.stack(traversable_pixels, dim=1).float()
                    xy_map_t = th.as_tensor(xy_map, dtype=th.float32)
                    dists = th.norm(pixels - xy_map_t.reshape(1, 2), dim=1)
                    idx = int(th.argmin(dists).item())
                    nearest_map = pixels[idx].int()
                    nearest_world = trav_map.map_to_world(nearest_map)
                    return {
                        "map": [int(nearest_map[0].item()), int(nearest_map[1].item())],
                        "world": _debug_array_value(nearest_world),
                        "pixel_dist": float(dists[idx].item()),
                    }
                except Exception as e:
                    return {"error": f"{type(e).__name__}: {e}"}

            def _pose_trav_record(label, pose_xy, trav_map, raw_map, eroded_map, labels, floor):
                record = {"label": label, "pose_xy": _debug_array_value(pose_xy)}
                try:
                    pose_xy_np = _debug_to_np(pose_xy)
                    if pose_xy_np is None or pose_xy_np.shape[0] < 2 or not bool(np.isfinite(pose_xy_np[:2]).all()):
                        record["error"] = "invalid_pose_xy"
                        return record

                    xy_map = trav_map.world_to_map(th.as_tensor(pose_xy_np[:2], dtype=th.float32))
                    row = int(xy_map[0].item())
                    col = int(xy_map[1].item())
                    shape = list(raw_map.shape)
                    in_bounds = 0 <= row < shape[0] and 0 <= col < shape[1]
                    record.update(
                        {
                            "floor": int(floor),
                            "map_xy": [row, col],
                            "map_shape": shape,
                            "in_bounds": bool(in_bounds),
                        }
                    )
                    if in_bounds:
                        raw_value = raw_map[row, col]
                        eroded_value = eroded_map[row, col]
                        record.update(
                            {
                                "raw_value": int(raw_value.item() if hasattr(raw_value, "item") else raw_value),
                                "eroded_value": int(eroded_value.item() if hasattr(eroded_value, "item") else eroded_value),
                                "raw_traversable": bool(raw_value != 0),
                                "eroded_traversable": bool(eroded_value != 0),
                                "component_label": int(labels[row, col]),
                            }
                        )
                    else:
                        record.update(
                            {
                                "raw_value": None,
                                "eroded_value": None,
                                "raw_traversable": False,
                                "eroded_traversable": False,
                                "component_label": None,
                            }
                        )
                    if not record["eroded_traversable"]:
                        record["nearest_eroded_traversable"] = _nearest_traversable(trav_map, eroded_map, [row, col])
                    try:
                        record["room"] = robot.scene._seg_map.get_room_instance_by_point(pose_xy_np[:2])
                    except Exception as e:
                        record["room"] = f"ERR:{type(e).__name__}: {e}"
                    return record
                except Exception as e:
                    record["error"] = f"{type(e).__name__}: {e}"
                    return record

            try:
                trav_map = getattr(robot.scene, "trav_map", None) or getattr(robot.scene, "_trav_map", None)
                if trav_map is None or getattr(trav_map, "floor_map", None) is None:
                    return _log_nav_debug(stage, traversability={"error": "trav_map_unavailable"})

                robot_base_pos = robot.get_position_orientation()[0]
                floor = _floor_for_z(_debug_to_np(robot_base_pos)[2], trav_map)
                raw_map = th.clone(trav_map.floor_map[floor])
                eroded_map = trav_map._erode_trav_map(th.clone(raw_map), robot=robot)

                import cv2
                from omnigibson.utils.motion_planning_utils import astar

                _, labels = cv2.connectedComponents(eroded_map.cpu().numpy(), connectivity=4)
                start_record = _pose_trav_record("current_base", robot_base_pos[:2], trav_map, raw_map, eroded_map, labels, floor)
                target_record = None
                path_record = None
                if target_pose_2d is not None:
                    target_np = _debug_to_np(target_pose_2d)
                    target_record = _pose_trav_record(
                        "target_base", None if target_np is None else target_np[:2], trav_map, raw_map, eroded_map, labels, floor
                    )
                    same_component = (
                        start_record.get("component_label") is not None
                        and target_record.get("component_label") is not None
                        and start_record.get("component_label") != 0
                        and start_record.get("component_label") == target_record.get("component_label")
                    )
                    path_record = {"same_component": bool(same_component)}
                    if start_record.get("in_bounds") and target_record.get("in_bounds"):
                        source_map = tuple(start_record["map_xy"])
                        target_map = tuple(target_record["map_xy"])
                        path_map = astar(eroded_map, source_map, target_map)
                        path_record.update(
                            {
                                "astar_found": path_map is not None,
                                "astar_num_points": None if path_map is None else int(len(path_map)),
                            }
                        )
                    else:
                        path_record.update({"astar_found": False, "astar_num_points": None, "astar_skipped": "out_of_bounds"})

                candidate_records = []
                if candidate_poses is not None:
                    for idx, pose in enumerate(candidate_poses):
                        pose_np = _debug_to_np(pose)
                        candidate_records.append(
                            _pose_trav_record(
                                f"candidate_{idx}", None if pose_np is None else pose_np[:2], trav_map, raw_map, eroded_map, labels, floor
                            )
                        )

                return _log_nav_debug(
                    stage,
                    traversability={
                        "floor": int(floor),
                        "map_resolution": float(getattr(trav_map, "map_resolution", np.nan)),
                        "robot_erosion_radius_source": "robot.reset_joint_pos_aabb_extent",
                        "start": start_record,
                        "target": target_record,
                        "path": path_record,
                        "candidates": candidate_records,
                    },
                )
            except Exception as e:
                return _log_nav_debug(stage, traversability={"error": f"{type(e).__name__}: {e}"})

        def _install_nav_debug_wrappers(selected_nav_arm, eef_pose_arg):
            nav_debug_enabled = bool(int(os.environ.get("MOMAGEN_NAV_DEBUG", "0") or 0))
            trav_debug_enabled = bool(int(os.environ.get("MOMAGEN_DEBUG_TRAV", "0") or 0))
            curobo_debug_enabled = bool(int(os.environ.get("MOMAGEN_DEBUG_CUROBO_BASE", "0") or 0))
            ignore_attached_obj_for_base_diag = bool(
                int(os.environ.get("MOMAGEN_NAV_IGNORE_ATTACHED_OBJ_FOR_BASE_DIAG", "0") or 0)
            )
            nav_arm_mp_feasibility_enabled = _nav_arm_mp_feasibility_filter_enabled()
            source_base_pose_enabled = bool(int(os.environ.get("MOMAGEN_NAV_TRY_SOURCE_BASE_POSE", "0") or 0))
            source_base_pose_min_phase = int(os.environ.get("MOMAGEN_NAV_SOURCE_BASE_POSE_MIN_PHASE", "0") or 0)
            source_base_pose_max_phase = int(os.environ.get("MOMAGEN_NAV_SOURCE_BASE_POSE_MAX_PHASE", "999999") or 999999)
            source_base_pose_enabled = (
                source_base_pose_enabled
                and source_base_pose_min_phase <= int(env.execution_phase_ind) <= source_base_pose_max_phase
            )
            if (
                not nav_debug_enabled
                and not trav_debug_enabled
                and not curobo_debug_enabled
                and not ignore_attached_obj_for_base_diag
                and not nav_arm_mp_feasibility_enabled
                and not source_base_pose_enabled
            ):
                return None

            primitive = env.primitive
            motion_generator = primitive._motion_generator
            originals = {
                "sample": primitive._sample_pose_near_object,
                "validate": primitive._validate_poses,
                "plan": primitive._plan_joint_motion,
                "check_collisions": getattr(motion_generator, "check_collisions", None),
                "compute_trajectories": getattr(motion_generator, "compute_trajectories", None),
                "plan_batch": getattr(motion_generator, "plan_batch", None),
                "solve_ik_batch": getattr(motion_generator, "solve_ik_batch", None),
            }

            _log_nav_debug(
                "pre_navigation_trial",
                base_mp_trial=int(base_mp_trial),
                **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
            )
            eef_aligned_base_pose_consumed = {"value": False}
            source_base_pose_consumed = {"value": False}

            def _sample_pose_near_object_debug(*args, **kwargs):
                _log_nav_debug(
                    "sample_pose_start",
                    base_mp_trial=int(base_mp_trial),
                    args_obj_names=[getattr(arg, "name", str(arg)) for arg in args[:1]],
                    eef_pose_kw=_debug_pose_value(kwargs.get("eef_pose")),
                    sampling_attempts=kwargs.get("sampling_attempts"),
                    skip_obstacle_update=kwargs.get("skip_obstacle_update"),
                    **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
                )
                try:
                    source_base_pose_every_trial = bool(
                        int(os.environ.get("MOMAGEN_NAV_SOURCE_BASE_POSE_EVERY_TRIAL", "0") or 0)
                    )
                    if (
                        source_base_pose_enabled
                        and not source_base_pose_consumed["value"]
                        and (base_mp_trial == 0 or source_base_pose_every_trial)
                    ):
                        source_base_pose_consumed["value"] = True
                        source_pose_2d, source_base_idx = _source_base_pose_for_nav_arm(selected_nav_arm)
                        if source_pose_2d is not None:
                            original_source_pose_2d = source_pose_2d
                            source_pose_2d, source_snap_record = _snap_pose_to_current_traversable_component(source_pose_2d)
                            _log_nav_debug(
                                "sample_pose_source_base_result",
                                base_mp_trial=int(base_mp_trial),
                                source_base_idx=None if source_base_idx is None else int(source_base_idx),
                                sampled_pose_2d=_debug_array_value(source_pose_2d),
                                original_sampled_pose_2d=_debug_array_value(original_source_pose_2d),
                                source_snap=source_snap_record,
                                selected_nav_arm=selected_nav_arm,
                            )
                            _log_traversability_debug("sample_pose_source_base_traversability", target_pose_2d=source_pose_2d)
                            return source_pose_2d
                        _log_nav_debug(
                            "sample_pose_source_base_unavailable",
                            base_mp_trial=int(base_mp_trial),
                            source_base_idx=None if source_base_idx is None else int(source_base_idx),
                            selected_nav_arm=selected_nav_arm,
                            has_src_curr_phase_base_pose=src_curr_phase_base_pose is not None,
                        )
                    if (
                        nav_arm_mp_feasibility_enabled
                        and bool(int(os.environ.get("MOMAGEN_NAV_TRY_EEF_ALIGNED_BASE_POSE", "0") or 0))
                        and not eef_aligned_base_pose_consumed["value"]
                    ):
                        probe_emb_sel = getattr(
                            CuRoboEmbodimentSelection,
                            "ARM_NO_TORSO",
                            CuRoboEmbodimentSelection.ARM,
                        )
                        if probe_emb_sel not in getattr(env.cmg, "mg", {}):
                            probe_emb_sel = CuRoboEmbodimentSelection.ARM
                        aligned_pose = _eef_aligned_base_pose(eef_pose_arg, selected_nav_arm, probe_emb_sel)
                        eef_aligned_base_pose_consumed["value"] = True
                        if aligned_pose is not None and bool(th.isfinite(aligned_pose).all()):
                            _log_nav_debug(
                                "sample_pose_eef_aligned_base_result",
                                base_mp_trial=int(base_mp_trial),
                                sampled_pose_2d=_debug_array_value(aligned_pose),
                                emb_sel=str(probe_emb_sel),
                            )
                            _log_traversability_debug("sample_pose_eef_aligned_base_traversability", target_pose_2d=aligned_pose)
                            return aligned_pose
                    result = originals["sample"](*args, **kwargs)
                    candidate_room = None
                    if result is not None:
                        try:
                            candidate_room = robot.scene._seg_map.get_room_instance_by_point(result[:2])
                        except Exception as e:
                            candidate_room = f"ERR:{type(e).__name__}: {e}"
                    _log_nav_debug(
                        "sample_pose_result",
                        base_mp_trial=int(base_mp_trial),
                        sampled_pose_2d=_debug_array_value(result),
                        sampled_pose_room=candidate_room,
                    )
                    _log_traversability_debug("sample_pose_traversability", target_pose_2d=result)
                    return result
                except Exception as e:
                    _log_nav_debug(
                        "sample_pose_exception",
                        base_mp_trial=int(base_mp_trial),
                        error=f"{type(e).__name__}: {e}",
                    )
                    raise

            def _validate_poses_debug(candidate_poses, *args, **kwargs):
                try:
                    result = originals["validate"](candidate_poses, *args, **kwargs)
                    result = _apply_nav_arm_mp_feasibility_filter(
                        result,
                        candidate_poses,
                        eef_pose_arg=eef_pose_arg,
                        plan_with_open_gripper=kwargs.get("plan_with_open_gripper", False),
                        skip_obstacle_update=kwargs.get("skip_obstacle_update", False),
                        compute_trajectories_fn=originals.get("compute_trajectories"),
                    )
                    if nav_debug_enabled or trav_debug_enabled:
                        _log_nav_debug(
                            "validate_poses_result",
                            base_mp_trial=int(base_mp_trial),
                            candidate_poses=[_debug_array_value(pose) for pose in candidate_poses],
                            result=_debug_array_value(result),
                            eef_pose_kw=_debug_pose_value(kwargs.get("eef_pose")),
                            skip_obstacle_update=kwargs.get("skip_obstacle_update"),
                            plan_with_open_gripper=kwargs.get("plan_with_open_gripper"),
                            **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
                        )
                        _log_traversability_debug("validate_poses_traversability", candidate_poses=candidate_poses)
                    return result
                except Exception as e:
                    _log_nav_debug(
                        "validate_poses_exception",
                        base_mp_trial=int(base_mp_trial),
                        candidate_poses=[_debug_array_value(pose) for pose in candidate_poses],
                        error=f"{type(e).__name__}: {e}",
                    )
                    raise

            def _plan_joint_motion_debug(*args, **kwargs):
                target_pos = kwargs.get("target_pos", args[0] if len(args) > 0 else None)
                target_quat = kwargs.get("target_quat", args[1] if len(args) > 1 else None)
                embodiment_selection = kwargs.get(
                    "embodiment_selection", args[2] if len(args) > 2 else CuRoboEmbodimentSelection.DEFAULT
                )
                _log_nav_debug(
                    "plan_joint_motion_start",
                    base_mp_trial=int(base_mp_trial),
                    target_pos={str(k): _debug_array_value(v) for k, v in (target_pos or {}).items()},
                    target_quat={str(k): _debug_array_value(v) for k, v in (target_quat or {}).items()},
                    embodiment_selection=str(embodiment_selection),
                    skip_obstacle_update=kwargs.get("skip_obstacle_update"),
                    ignore_objects=[getattr(obj, "name", str(obj)) for obj in (kwargs.get("ignore_objects") or [])],
                    **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
                )
                try:
                    base_target_pose = None
                    if str(embodiment_selection) == str(CuRoboEmbodimentSelection.BASE) and target_pos:
                        base_link_name = getattr(robot, "base_footprint_link_name", None)
                        base_target_pos = target_pos.get(base_link_name) if base_link_name in target_pos else next(iter(target_pos.values()))
                        base_target_np = _debug_to_np(base_target_pos)
                        base_target_pose = None if base_target_np is None else base_target_np[:2]
                    _log_traversability_debug("plan_joint_motion_traversability", target_pose_2d=base_target_pose)
                except Exception as e:
                    _log_nav_debug("plan_joint_motion_traversability", traversability={"error": f"{type(e).__name__}: {e}"})
                try:
                    ignore_attached_obj_for_this_plan = (
                        ignore_attached_obj_for_base_diag
                        and str(embodiment_selection) == str(CuRoboEmbodimentSelection.BASE)
                    )
                    original_get_obj_in_hand = None
                    if ignore_attached_obj_for_this_plan:
                        original_get_obj_in_hand = primitive._get_obj_in_hand
                        try:
                            obj_in_hand = original_get_obj_in_hand()
                        except Exception as obj_e:
                            obj_in_hand = f"ERR:{type(obj_e).__name__}: {obj_e}"
                        _log_nav_debug(
                            "plan_joint_motion_ignore_attached_obj_for_base_diag",
                            base_mp_trial=int(base_mp_trial),
                            ignored_obj_in_hand=(
                                getattr(obj_in_hand, "name", str(obj_in_hand))
                                if obj_in_hand is not None
                                else None
                            ),
                            ignored_obj_in_hand_root_link=(
                                getattr(getattr(obj_in_hand, "root_link", None), "name", None)
                                if not isinstance(obj_in_hand, str)
                                else None
                            ),
                            embodiment_selection=str(embodiment_selection),
                            **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
                        )
                        primitive._get_obj_in_hand = lambda: None
                    try:
                        result = originals["plan"](*args, **kwargs)
                    finally:
                        if original_get_obj_in_hand is not None:
                            primitive._get_obj_in_hand = original_get_obj_in_hand
                    _log_nav_debug(
                        "plan_joint_motion_result",
                        base_mp_trial=int(base_mp_trial),
                        q_traj_shape=list(result.shape) if hasattr(result, "shape") else None,
                        base_planner_sources=getattr(motion_generator, "_last_base_joint_plan_sources", None),
                    )
                    return result
                except Exception as e:
                    _log_nav_debug(
                        "plan_joint_motion_exception",
                        base_mp_trial=int(base_mp_trial),
                        error=f"{type(e).__name__}: {e}",
                    )
                    raise

            def _base_goal_q_from_target(target_pos, target_quat):
                try:
                    if not isinstance(target_pos, dict) or not target_pos:
                        return None
                    base_link_name = getattr(robot, "base_footprint_link_name", None)
                    base_target_pos = target_pos.get(base_link_name) if base_link_name in target_pos else next(iter(target_pos.values()))
                    base_target_quat = (
                        target_quat.get(base_link_name) if isinstance(target_quat, dict) and base_link_name in target_quat
                        else next(iter(target_quat.values())) if isinstance(target_quat, dict) and target_quat else None
                    )
                    base_target_pos_np = _debug_to_np(base_target_pos)
                    base_target_quat_t = th.as_tensor(base_target_quat[0] if getattr(base_target_quat, "dim", lambda: 0)() == 2 else base_target_quat)
                    base_target_yaw = float(T.quat2euler(base_target_quat_t)[2].item()) if base_target_quat is not None else None
                    if base_target_pos_np is None or base_target_pos_np.shape[-1] < 2 or base_target_yaw is None:
                        return None
                    if base_target_pos_np.ndim > 1:
                        base_target_pos_np = base_target_pos_np[0]

                    q_goal = robot.get_joint_positions().clone()
                    base_control_idx = th.as_tensor(robot.base_control_idx).long()
                    if len(base_control_idx) >= 3:
                        q_goal[base_control_idx[0]] = float(base_target_pos_np[0])
                        q_goal[base_control_idx[1]] = float(base_target_pos_np[1])
                        q_goal[base_control_idx[2]] = base_target_yaw
                    return q_goal
                except Exception:
                    return None

            def _get_curobo_base_debug_record(target_pos=None, target_quat=None, is_local=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
                record = {"enabled": bool(curobo_debug_enabled), "embodiment_selection": str(emb_sel)}
                try:
                    mg = motion_generator
                    emb_key = emb_sel
                    mg_obj = getattr(mg, "mg", {}).get(emb_key)
                    record.update(
                        {
                            "mg_present": mg_obj is not None,
                            "batch_size": getattr(mg, "batch_size", None),
                            "robot_joint_names_count": len(getattr(mg, "robot_joint_names", []) or []),
                            "robot_joint_names_head": list((getattr(mg, "robot_joint_names", []) or [])[:8]),
                            "robot_base_joint_names": list(getattr(robot, "base_joint_names", []) or []),
                            "robot_base_control_idx": _debug_array_value(getattr(robot, "base_control_idx", None)),
                            "robot_base_idx": _debug_array_value(getattr(robot, "base_idx", None)),
                            "robot_base_footprint_link_name": getattr(robot, "base_footprint_link_name", None),
                            "curobo_base_link": getattr(mg, "base_link", {}).get(emb_key),
                            "curobo_ee_link": getattr(mg, "ee_link", {}).get(emb_key),
                            "curobo_additional_links": list(getattr(mg, "additional_links", {}).get(emb_key, []) or []),
                        }
                    )
                    if mg_obj is not None:
                        kin = getattr(mg_obj, "kinematics", None)
                        kin_config = getattr(kin, "kinematics_config", None)
                        joint_names = list(getattr(kin, "joint_names", []) or [])
                        record["curobo_kinematics_joint_names"] = joint_names
                        lock_jointstate = getattr(kin_config, "lock_jointstate", None)
                        record["lock_joint_names"] = list(getattr(lock_jointstate, "joint_names", []) or [])
                        record["lock_joint_position"] = _debug_array_value(getattr(lock_jointstate, "position", None))
                        joint_limits = getattr(kin_config, "joint_limits", None)
                        limit_names = list(getattr(joint_limits, "joint_names", []) or [])
                        limit_pos = getattr(joint_limits, "position", None)
                        base_limits = {}
                        for joint_name in getattr(robot, "base_joint_names", []) or []:
                            if joint_name in limit_names:
                                idx = limit_names.index(joint_name)
                                try:
                                    base_limits[joint_name] = [
                                        float(limit_pos[0][idx].item()),
                                        float(limit_pos[1][idx].item()),
                                    ]
                                except Exception as e:
                                    base_limits[joint_name] = f"ERR:{type(e).__name__}: {e}"
                        record["base_joint_limits"] = base_limits

                    current_q = robot.get_joint_positions()
                    record["current_q"] = _debug_array_value(current_q)
                    record["current_base_control_q"] = _debug_array_value(current_q[th.as_tensor(robot.base_control_idx).long()])
                    q_goal = _base_goal_q_from_target(target_pos, target_quat)
                    record["goal_q"] = _debug_array_value(q_goal)
                    if q_goal is not None:
                        record["goal_base_control_q"] = _debug_array_value(q_goal[th.as_tensor(robot.base_control_idx).long()])

                    if isinstance(target_pos, dict) and target_pos:
                        base_link_name = getattr(robot, "base_footprint_link_name", None)
                        target_key = base_link_name if base_link_name in target_pos else next(iter(target_pos.keys()))
                        target_pos_t = target_pos[target_key]
                        target_quat_t = target_quat[target_key] if isinstance(target_quat, dict) and target_key in target_quat else None
                        if getattr(target_pos_t, "dim", lambda: 0)() == 2:
                            target_pos_first = target_pos_t[0]
                        else:
                            target_pos_first = target_pos_t
                        if target_quat_t is not None and getattr(target_quat_t, "dim", lambda: 0)() == 2:
                            target_quat_first = target_quat_t[0]
                        else:
                            target_quat_first = target_quat_t
                        record["world_target_link"] = target_key
                        record["world_target_pos_first"] = _debug_array_value(target_pos_first)
                        record["world_target_quat_first_xyzw"] = _debug_array_value(target_quat_first)

                        if not is_local and target_quat_first is not None:
                            try:
                                curobo_base_link_name = getattr(mg, "base_link", {}).get(emb_key)
                                robot_pos, robot_quat = robot.links[curobo_base_link_name].get_position_orientation()
                                target_pose = th.zeros((4, 4), dtype=target_pos_first.dtype)
                                target_pose[3, 3] = 1.0
                                target_pose[:3, :3] = T.quat2mat(target_quat_first)
                                target_pose[:3, 3] = target_pos_first
                                inv_robot_pose = T.pose_inv(T.pose2mat((robot_pos, robot_quat)))
                                target_pose_local = inv_robot_pose @ target_pose
                                local_pos = target_pose_local[:3, 3]
                                local_quat_xyzw = T.mat2quat(target_pose_local[:3, :3])
                                record["curobo_base_link_pose"] = _debug_pose_record(
                                    robot.links[curobo_base_link_name].get_position_orientation,
                                    "curobo_base_link_pose",
                                    [],
                                )
                                record["target_pos_in_curobo_base_frame"] = _debug_array_value(local_pos)
                                record["target_quat_in_curobo_base_frame_xyzw"] = _debug_array_value(local_quat_xyzw)
                                record["target_quat_in_curobo_base_frame_wxyz"] = _debug_array_value(local_quat_xyzw[[3, 0, 1, 2]])
                            except Exception as e:
                                record["target_frame_transform_error"] = f"{type(e).__name__}: {e}"

                    if originals.get("check_collisions") is not None and q_goal is not None:
                        try:
                            q_check = th.stack([current_q, q_goal], dim=0)
                            collision_result = originals["check_collisions"](
                                q_check,
                                initial_joint_pos=current_q,
                                self_collision_check=False,
                                skip_obstacle_update=True,
                                attached_obj=None,
                            )
                            record["default_world_collision_current_goal_no_attached"] = _debug_array_value(collision_result)
                        except Exception as e:
                            record["default_world_collision_current_goal_no_attached_error"] = f"{type(e).__name__}: {e}"
                        try:
                            obj_in_hand = primitive._get_obj_in_hand()
                            attached_obj = (
                                {robot.eef_link_names[getattr(primitive, "arm", selected_nav_arm)]: obj_in_hand.root_link}
                                if obj_in_hand is not None and getattr(primitive, "arm", None) in robot.eef_link_names
                                else None
                            )
                            collision_result = originals["check_collisions"](
                                th.stack([current_q, q_goal], dim=0),
                                initial_joint_pos=current_q,
                                self_collision_check=False,
                                skip_obstacle_update=True,
                                attached_obj=attached_obj,
                            )
                            record["default_world_collision_current_goal_with_primitive_attached"] = _debug_array_value(collision_result)
                            record["primitive_attached_obj_for_probe"] = (
                                None if attached_obj is None else {k: getattr(v, "name", str(v)) for k, v in attached_obj.items()}
                            )
                        except Exception as e:
                            record["default_world_collision_current_goal_with_primitive_attached_error"] = f"{type(e).__name__}: {e}"

                    return record
                except Exception as e:
                    record["error"] = f"{type(e).__name__}: {e}"
                    return record

            def _compute_trajectories_debug(*args, **kwargs):
                emb_sel = kwargs.get("emb_sel", CuRoboEmbodimentSelection.DEFAULT)
                is_base = str(emb_sel) == str(CuRoboEmbodimentSelection.BASE)
                if curobo_debug_enabled and is_base:
                    _log_nav_debug(
                        "curobo_compute_trajectories_start",
                        base_mp_trial=int(base_mp_trial),
                        target_pos={str(k): _debug_array_value(v) for k, v in (kwargs.get("target_pos") or {}).items()},
                        target_quat={str(k): _debug_array_value(v) for k, v in (kwargs.get("target_quat") or {}).items()},
                        is_local=kwargs.get("is_local"),
                        skip_obstacle_update=kwargs.get("skip_obstacle_update"),
                        max_attempts=kwargs.get("max_attempts"),
                        timeout=kwargs.get("timeout"),
                        ik_fail_return=kwargs.get("ik_fail_return"),
                        attached_obj=(
                            None if kwargs.get("attached_obj") is None
                            else {str(k): getattr(v, "name", str(v)) for k, v in kwargs.get("attached_obj", {}).items()}
                        ),
                        curobo_base_probe=_get_curobo_base_debug_record(
                            target_pos=kwargs.get("target_pos"),
                            target_quat=kwargs.get("target_quat"),
                            is_local=kwargs.get("is_local", False),
                            emb_sel=emb_sel,
                        ),
                    )
                try:
                    result = originals["compute_trajectories"](*args, **kwargs)
                    if curobo_debug_enabled and is_base:
                        if isinstance(result, tuple):
                            successes, traj_paths = result
                            _log_nav_debug(
                                "curobo_compute_trajectories_result",
                                base_mp_trial=int(base_mp_trial),
                                successes=_debug_array_value(successes),
                                num_paths=(None if traj_paths is None else len(traj_paths)),
                                none_path_count=(
                                    None if traj_paths is None else sum(1 for path in traj_paths if path is None)
                                ),
                            )
                        else:
                            result_records = []
                            for item in result:
                                result_records.append(
                                    {
                                        "success": _debug_array_value(getattr(item, "success", None)),
                                        "status": str(getattr(item, "status", None)),
                                        "interpolated_plan_shape": (
                                            None if getattr(item, "interpolated_plan", None) is None
                                            else list(item.interpolated_plan.shape)
                                        ),
                                    }
                                )
                            _log_nav_debug("curobo_compute_trajectories_result", base_mp_trial=int(base_mp_trial), results=result_records)
                    return result
                except Exception as e:
                    if curobo_debug_enabled and is_base:
                        _log_nav_debug(
                            "curobo_compute_trajectories_exception",
                            base_mp_trial=int(base_mp_trial),
                            error=f"{type(e).__name__}: {e}",
                        )
                    raise

            def _plan_batch_debug(start_state, goal_pose, plan_config, link_poses=None, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
                is_base = str(emb_sel) == str(CuRoboEmbodimentSelection.BASE)
                if curobo_debug_enabled and is_base:
                    _log_nav_debug(
                        "curobo_plan_batch_start",
                        base_mp_trial=int(base_mp_trial),
                        start_joint_names=list(getattr(start_state, "joint_names", []) or []),
                        start_position=_debug_array_value(getattr(start_state, "position", None)),
                        goal_name=getattr(goal_pose, "name", None),
                        goal_position=_debug_array_value(getattr(goal_pose, "position", None)),
                        goal_quaternion_wxyz=_debug_array_value(getattr(goal_pose, "quaternion", None)),
                        link_pose_names=None if link_poses is None else list(link_poses.keys()),
                        plan_config={
                            "max_attempts": getattr(plan_config, "max_attempts", None),
                            "timeout": getattr(plan_config, "timeout", None),
                            "enable_graph": getattr(plan_config, "enable_graph", None),
                            "enable_finetune_trajopt": getattr(plan_config, "enable_finetune_trajopt", None),
                            "success_ratio": getattr(plan_config, "success_ratio", None),
                        },
                    )
                result, success, joint_state = originals["plan_batch"](
                    start_state, goal_pose, plan_config, link_poses=link_poses, emb_sel=emb_sel
                )
                if curobo_debug_enabled and is_base:
                    _log_nav_debug(
                        "curobo_plan_batch_result",
                        base_mp_trial=int(base_mp_trial),
                        success=_debug_array_value(success),
                        result_success=_debug_array_value(getattr(result, "success", None)),
                        result_status=str(getattr(result, "status", None)),
                        interpolated_plan_shape=(
                            None if getattr(result, "interpolated_plan", None) is None else list(result.interpolated_plan.shape)
                        ),
                        num_joint_state_paths=(None if joint_state is None else len(joint_state)),
                        none_joint_state_count=(None if joint_state is None else sum(1 for path in joint_state if path is None)),
                    )
                return result, success, joint_state

            def _solve_ik_batch_debug(start_state, goal_pose, plan_config, link_poses=None, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
                result, success, joint_state = originals["solve_ik_batch"](
                    start_state, goal_pose, plan_config, link_poses=link_poses, emb_sel=emb_sel
                )
                if curobo_debug_enabled and str(emb_sel) == str(CuRoboEmbodimentSelection.BASE):
                    _log_nav_debug(
                        "curobo_solve_ik_batch_result",
                        base_mp_trial=int(base_mp_trial),
                        success=_debug_array_value(success),
                        result_success=_debug_array_value(getattr(result, "success", None)),
                        result_status=str(getattr(result, "status", None)),
                    )
                return result, success, joint_state

            def _check_collisions_debug(q, *args, **kwargs):
                attached_obj = kwargs.get("attached_obj")
                attached_summary = None
                if attached_obj is not None:
                    attached_summary = {
                        str(link_name): getattr(root_link, "name", str(root_link))
                        for link_name, root_link in attached_obj.items()
                    }
                _log_nav_debug(
                    "check_collisions_start",
                    base_mp_trial=int(base_mp_trial),
                    q=_debug_array_value(q),
                    self_collision_check=kwargs.get("self_collision_check"),
                    skip_obstacle_update=kwargs.get("skip_obstacle_update"),
                    attached_obj=attached_summary,
                    attached_obj_scale=kwargs.get("attached_obj_scale"),
                    attached_obj_options=kwargs.get("attached_obj_options"),
                    **_get_nav_debug_robot_state(selected_nav_arm=selected_nav_arm, eef_pose_arg=eef_pose_arg),
                )
                try:
                    result = originals["check_collisions"](q, *args, **kwargs)
                    result_np = _debug_to_np(result)
                    candidate_diag = None
                    if (
                        bool(int(os.environ.get("MOMAGEN_NAV_COLLISION_CANDIDATE_DIAG", "0") or 0))
                        and attached_obj is not None
                        and getattr(q, "dim", lambda: 0)() == 2
                    ):
                        candidate_diag = []
                        base_control_idx_t = th.as_tensor(robot.base_control_idx).long()
                        initial_joint_pos = kwargs.get("initial_joint_pos")
                        if initial_joint_pos is None:
                            initial_joint_pos = robot.get_joint_positions()
                        for candidate_idx in range(int(q.shape[0])):
                            row = q[candidate_idx]
                            record = {
                                "candidate_index": int(candidate_idx),
                                "base_control_q": _debug_array_value(row[base_control_idx_t]),
                            }
                            try:
                                no_attached_result = originals["check_collisions"](
                                    row.unsqueeze(0),
                                    *args,
                                    **{**kwargs, "attached_obj": None},
                                )
                                record["no_attached_collision"] = _debug_array_value(no_attached_result)
                            except Exception as e:
                                record["no_attached_collision_error"] = f"{type(e).__name__}: {e}"

                            attached_obj_options = {}
                            for link_name in attached_obj.keys():
                                candidate_link_pos, candidate_link_quat = _candidate_robot_link_pose_after_base_move(
                                    row[base_control_idx_t].detach().cpu(),
                                    link_name,
                                    CuRoboEmbodimentSelection.DEFAULT,
                                )
                                if candidate_link_pos is not None and candidate_link_quat is not None:
                                    attached_obj_options[link_name] = {
                                        "ee_pose": (candidate_link_pos, candidate_link_quat),
                                    }
                            attached_obj_options = _attached_payload_options(
                                attached_obj,
                                base_options=attached_obj_options,
                            )
                            if attached_obj_options:
                                try:
                                    candidate_attached_result = originals["check_collisions"](
                                        row.unsqueeze(0),
                                        *args,
                                        **{
                                            **kwargs,
                                            "initial_joint_pos": initial_joint_pos,
                                            "attached_obj_options": attached_obj_options,
                                        },
                                    )
                                    record["candidate_ee_pose_attached_collision"] = _debug_array_value(
                                        candidate_attached_result
                                    )
                                    record["candidate_attached_ee_pose"] = {
                                        str(link_name): {
                                            "pos": _debug_array_value(options["ee_pose"][0]),
                                            "quat": _debug_array_value(options["ee_pose"][1]),
                                        }
                                        for link_name, options in attached_obj_options.items()
                                    }
                                except Exception as e:
                                    record["candidate_ee_pose_attached_collision_error"] = f"{type(e).__name__}: {e}"
                            candidate_diag.append(record)
                    _log_nav_debug(
                        "check_collisions_result",
                        base_mp_trial=int(base_mp_trial),
                        result=_debug_array_value(result),
                        invalid_count=(None if result_np is None else int(np.asarray(result_np).astype(bool).sum())),
                        attached_obj=attached_summary,
                        candidate_diag=candidate_diag,
                    )
                    return result
                except Exception as e:
                    _log_nav_debug(
                        "check_collisions_exception",
                        base_mp_trial=int(base_mp_trial),
                        error=f"{type(e).__name__}: {e}",
                        attached_obj=attached_summary,
                    )
                    raise

            if nav_debug_enabled or trav_debug_enabled or nav_arm_mp_feasibility_enabled:
                primitive._sample_pose_near_object = _sample_pose_near_object_debug
            primitive._validate_poses = _validate_poses_debug
            if nav_debug_enabled or trav_debug_enabled or curobo_debug_enabled or ignore_attached_obj_for_base_diag:
                primitive._plan_joint_motion = _plan_joint_motion_debug
            if curobo_debug_enabled and originals["compute_trajectories"] is not None:
                motion_generator.compute_trajectories = _compute_trajectories_debug
            if curobo_debug_enabled and originals["plan_batch"] is not None:
                motion_generator.plan_batch = _plan_batch_debug
            if curobo_debug_enabled and originals["solve_ik_batch"] is not None:
                motion_generator.solve_ik_batch = _solve_ik_batch_debug
            if (nav_debug_enabled or curobo_debug_enabled) and originals["check_collisions"] is not None:
                motion_generator.check_collisions = _check_collisions_debug
            return originals

        def _restore_nav_debug_wrappers(originals):
            if originals is None:
                return
            env.primitive._sample_pose_near_object = originals["sample"]
            env.primitive._validate_poses = originals["validate"]
            env.primitive._plan_joint_motion = originals["plan"]
            if originals.get("compute_trajectories") is not None:
                env.primitive._motion_generator.compute_trajectories = originals["compute_trajectories"]
            if originals.get("plan_batch") is not None:
                env.primitive._motion_generator.plan_batch = originals["plan_batch"]
            if originals.get("solve_ik_batch") is not None:
                env.primitive._motion_generator.solve_ik_batch = originals["solve_ik_batch"]
            if originals.get("check_collisions") is not None:
                env.primitive._motion_generator.check_collisions = originals["check_collisions"]

        # ================================= Base Navigation ==================================
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            seq = self.waypoint_sequences[0]

            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            left_mp_last_waypoint = left_mp_waypoints[-1]
            left_waypoints = [left_mp_last_waypoint] + left_replay_waypoints

            left_waypoint_pos = th.vstack([th.tensor(wp.pose[0:3, 3]) for wp in left_waypoints])
            left_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[0:3, 0:3])) for wp in left_waypoints])

            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]
            right_mp_last_waypoint = right_mp_waypoints[-1]
            right_waypoints = [right_mp_last_waypoint] + right_replay_waypoints

            right_waypoint_pos = th.vstack([th.tensor(wp.pose[4:7, 3]) for wp in right_waypoints])
            right_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[4:7, 0:3])) for wp in right_waypoints])

            left_waypoint_pos, right_waypoint_pos = self._pad_tensors(left_waypoint_pos, right_waypoint_pos)
            left_waypoint_ori, right_waypoint_ori = self._pad_tensors(left_waypoint_ori, right_waypoint_ori)

            left_waypoint_pos = self._subsample_tensor(left_waypoint_pos)
            left_waypoint_ori = self._subsample_tensor(left_waypoint_ori)
            right_waypoint_pos = self._subsample_tensor(right_waypoint_pos)
            right_waypoint_ori = self._subsample_tensor(right_waypoint_ori)

            # left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            # left_waypoint = left_mp_waypoints[-1]
            # left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            # right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            # right_waypoint = right_mp_waypoints[-1]
            # right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))

            # _navigate_to_obj expects a single target EEF pose per relevant arm,
            # not the full MP+replay waypoint sequence.  The first row is the MP
            # target that base navigation should bring into local arm reach; the
            # remaining rows are replay waypoints consumed later by the arm phase.
            eef_pose = {
                "left": (left_waypoint_pos[0], left_waypoint_ori[0]),
                "right": (right_waypoint_pos[0], right_waypoint_ori[0])
            }
            eef_pose, _ = maybe_apply_phase_routing_target_precontact(
                eef_pose,
                env=env,
                ref_obj=ref_obj,
                object_ref=object_ref,
                phase_type=phase_type,
                phase_logs=phase_logs,
            )

            if enable_marker_vis:
                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                # env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                # env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)
                env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos[0], orientation=left_waypoint_ori[0])
                env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos[0], orientation=right_waypoint_ori[0])
                for _ in range(10): og.sim.step()

            # TODO: Implement this
            check_torso_mode_first = False
            if check_torso_mode_first:
                pass
                # Given the ref obect and the eef poses, check if we can only move the torso to satisfy reachability and visibility
                # 0. attach object
                # 1. call _target_in_reach_of_robot_and_visible(self,
                #                                               eef_pose,
                #                                               initial_joint_pos=env.robot.get_joint_positions(),
                #                                               skip_obstacle_update=True,
                #                                               ik_world_collision_check=False,
                #                                               emb_sel=CuRoboEmbodimentSelection.ARM,
                #                                               attach_obj=False,
                #                                               eyes_pose=None):
                # So, above will sample eyes pose, check IK solving for (eef_poses, eyes_pose) w/o collision check, for samples that succeed previous Ik check
                # check if setting (current base + IK torso + current arms) is collision-free
                # 2. If yes, use the aforementioned (eyes_pose + eef poses) and do arm mode MP (which is with collision) and overwrite the arm actions to not do anything
                # 3. If above, succeeds, execute it
                # 4. else, continue to base MP


            num_tries = 3
            base_mp_trial = 0
            nav_mp_success = False
            while True:
                # Base condition
                if base_mp_trial == num_tries:
                    print("Base MP failed after {} trials. Giving up.".format(num_tries))
                    env.err = env.primitive.mp_err
                    # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase).
                    # In this case MP failed and phase was not actually executed
                    env.execution_phase_ind += 1
                    # env.valid_env = env.primitive.valid_env
                    return None

                print("Base MP trial: ", base_mp_trial)

                nav_arg_names = getattr(getattr(env.primitive, "_navigate_to_obj", None), "__code__", None)
                nav_arg_names = () if nav_arg_names is None else nav_arg_names.co_varnames
                enable_visibility_constraint = (
                    isinstance(env.robot, (R1, R1Pro))
                    and env.hard_visibility_constraint
                    and bool(int(os.environ.get("MOMAGEN_ENABLE_VISIBILITY_CONSTRAINT", "1") or 1))
                )
                if bool(int(os.environ.get("MOMAGEN_VISIBILITY_DEBUG", "0") or 0)):
                    print(
                        "[MOMAGEN_VISIBILITY_DEBUG] "
                        + json.dumps(
                            {
                                "phase": int(env.execution_phase_ind),
                                "base_mp_trial": int(base_mp_trial),
                                "robot_type": type(env.robot).__name__,
                                "hard_visibility_constraint": bool(env.hard_visibility_constraint),
                                "enable_visibility_constraint": bool(enable_visibility_constraint),
                                "primitive_accepts_visibility_constraint": "visibility_constraint" in nav_arg_names,
                            },
                            default=str,
                        ),
                        flush=True,
                    )

                def _navigate_to_obj_compatible(eef_pose_arg):
                    nav_skip_obstacle_update = bool(
                        int(os.environ.get("MOMAGEN_NAV_SKIP_OBSTACLE_UPDATE", "0") or 0)
                    )
                    # MoMaGen's intended flow is: if the manipulation target is not in
                    # local arm reach, first sample / plan a base pose using the target
                    # EEF pose, then run the local arm MP. Object-only navigation is a
                    # diagnostic fallback, not the default, because it can park the base
                    # at a pose that sees the object but still leaves the EEF target out
                    # of arm workspace.
                    if bool(int(os.environ.get("MOMAGEN_NAV_OBJECT_ONLY", "0") or 0)):
                        kwargs = {"obj": ref_obj}
                        if "skip_obstacle_update" in nav_arg_names:
                            kwargs["skip_obstacle_update"] = nav_skip_obstacle_update
                        return env.primitive._navigate_to_obj(**kwargs)
                    try:
                        kwargs = {
                            "obj": ref_obj,
                            "eef_pose": eef_pose_arg,
                            "visibility_constraint": enable_visibility_constraint,
                        }
                        if "skip_obstacle_update" in nav_arg_names:
                            kwargs["skip_obstacle_update"] = nav_skip_obstacle_update
                        return env.primitive._navigate_to_obj(**kwargs)
                    except TypeError as e:
                        if "visibility_constraint" not in str(e):
                            raise
                        try:
                            kwargs = {"obj": ref_obj, "eef_pose": eef_pose_arg}
                            if "skip_obstacle_update" in nav_arg_names:
                                kwargs["skip_obstacle_update"] = nav_skip_obstacle_update
                            return env.primitive._navigate_to_obj(**kwargs)
                        except Exception as inner_e:
                            if not bool(int(os.environ.get("MOMAGEN_NAV_FALLBACK_OBJECT_ONLY_ON_ERROR", "0") or 0)):
                                raise
                            print(
                                "EEF-pose navigation failed after visibility-constraint fallback; "
                                f"falling back to object-only navigation: {type(inner_e).__name__}: {inner_e}"
                            )
                            kwargs = {"obj": ref_obj}
                            if "skip_obstacle_update" in nav_arg_names:
                                kwargs["skip_obstacle_update"] = nav_skip_obstacle_update
                            return env.primitive._navigate_to_obj(**kwargs)
                    except Exception as e:
                        if not bool(int(os.environ.get("MOMAGEN_NAV_FALLBACK_OBJECT_ONLY_ON_ERROR", "0") or 0)):
                            raise
                        print(
                            "EEF-pose navigation failed; falling back to object-only navigation: "
                            f"{type(e).__name__}: {e}"
                        )
                        kwargs = {"obj": ref_obj}
                        if "skip_obstacle_update" in nav_arg_names:
                            kwargs["skip_obstacle_update"] = nav_skip_obstacle_update
                        return env.primitive._navigate_to_obj(**kwargs)

                # Pass only the eef that has a reference object associated with it (i.e. the arm that is relevant for this sub-step)
                nav_debug_originals = None
                if object_ref["arm_right"] is None:
                    selected_nav_arm = "left"
                elif object_ref["arm_left"] is None:
                    selected_nav_arm = "right"
                else:
                    selected_nav_arm = "both"
                eef_pose_arg = select_phase_routing_nav_eef_pose(eef_pose, selected_nav_arm)
                # OG's primitive exposes arm as a read-only view of robot.default_arm.  Do not mutate it here; the
                # BASE planner now detects attached objects across all gripper arms instead of relying on that
                # default-arm view.
                nav_debug_originals = _install_nav_debug_wrappers(selected_nav_arm, eef_pose_arg)
                try:
                    action_generator = _navigate_to_obj_compatible(eef_pose_arg)
                except Exception:
                    _restore_nav_debug_wrappers(nav_debug_originals)
                    nav_debug_originals = None
                    raise
                # action_generator = env.primitive._navigate_to_obj(obj=ref_obj, visibility_constraint=env.hard_visibility_constraint)

                init_state = og.sim.dump_state()
                pre_nav_ref_obj_pose = None
                if ref_obj is not None:
                    try:
                        pre_nav_ref_obj_pose = ref_obj.get_position_orientation()
                    except Exception:
                        pre_nav_ref_obj_pose = None
                local_env_step = 0
                states = []
                actions = []
                observations = []
                observations_info = []
                datagen_infos = []
                success = {"task": False}
                init_global_env_step = env.global_env_step
                # success = {k: False for k in env.is_success()} # success metrics
                nav_execution_start_time = time.time()
                try:
                    nav_exec_progress_interval = int(os.environ.get("MOMAGEN_NAV_EXEC_PROGRESS_INTERVAL", "0") or 0)
                    nav_exec_step_debug = bool(int(os.environ.get("MOMAGEN_NAV_EXEC_STEP_DEBUG", "0") or 0))
                    nav_exec_max_steps = int(os.environ.get("MOMAGEN_NAV_EXEC_MAX_STEPS", "0") or 0)
                    for temp_idx, mp_action in enumerate(action_generator):

                        # This will happen if
                        # 1. base sampling fails
                        # 2. base MP fails.
                        # 3. base execution fails to converge
                        if mp_action is None:
                            print(f"Base MP trial {base_mp_trial} failed. Retrying...")
                            base_mp_trial += 1
                            nav_mp_success = False
                            # This is there to avoid error in nav execution time (which in this case will always be 0)
                            nav_execution_start_time = time.time()
                            break
                        else:
                            nav_mp_success = True

                        if temp_idx == 0:
                            print("Time taken for base sampling: ", getattr(env.primitive, "base_sampling_time", 0))
                            print("Time taken for base MP planning: ", getattr(env.primitive, "base_mp_planning_time", 0))
                            nav_execution_start_time = time.time()

                        if nav_exec_progress_interval > 0 and temp_idx % nav_exec_progress_interval == 0:
                            try:
                                base_pose_dbg = env.robot.get_position_orientation()
                                print(
                                    "[MOMAGEN_NAV_EXEC_PROGRESS] "
                                    + json.dumps(
                                        {
                                            "phase": int(env.execution_phase_ind),
                                            "base_mp_trial": int(base_mp_trial),
                                            "step": int(temp_idx),
                                            "base_pos": _debug_to_np(base_pose_dbg[0]).tolist(),
                                            "base_quat": _debug_to_np(base_pose_dbg[1]).tolist(),
                                        },
                                        default=str,
                                    ),
                                    flush=True,
                                )
                            except Exception as progress_e:
                                print(f"[MOMAGEN_NAV_EXEC_PROGRESS] logging_failed={type(progress_e).__name__}: {progress_e}", flush=True)

                        if nav_exec_max_steps > 0 and temp_idx >= nav_exec_max_steps:
                            print(
                                f"Base MP trial {base_mp_trial} exceeded MOMAGEN_NAV_EXEC_MAX_STEPS={nav_exec_max_steps}. Retrying...",
                                flush=True,
                            )
                            env.primitive.mp_err = "BaseExecutionStepLimit"
                            base_mp_trial += 1
                            nav_mp_success = False
                            nav_execution_start_time = time.time()
                            break

                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=to_numpy_start", flush=True)
                        mp_action = mp_action.cpu().numpy()
                        # NOTE: For the MultiFinger gripper controler in binary mode that we use for tiago, we need to ensure that the
                        # gripper actions are correctly set based on whether an object is grasped by that gripper or not
                        if attached_obj["left"] is not None:
                            mp_action[robot.gripper_action_idx["left"]] = -1
                        if attached_obj["right"] is not None:
                            mp_action[robot.gripper_action_idx["right"]] = -1
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=get_state_start", flush=True)
                        state = env.get_state()["states"]
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=get_obs_start", flush=True)
                        obs, obs_info = env.get_obs_IL()
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=get_datagen_info_start", flush=True)
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        # print("mp_action[robot.base_action_idx]: ", mp_action[robot.base_action_idx])
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=env_step_start", flush=True)
                        env.step(mp_action, video_writer)
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=env_step_done", flush=True)
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=visibility_check_start", flush=True)
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        if nav_exec_step_debug:
                            print(f"[MOMAGEN_NAV_EXEC_STEP_DEBUG] step={temp_idx} stage=step_done", flush=True)
                except Exception as e:
                    print(f"Base MP trial {base_mp_trial} raised {type(e).__name__}: {e}. Retrying...")
                    if bool(int(os.environ.get("MOMAGEN_NAV_EXCEPTION_TRACEBACK", "0") or 0)):
                        traceback.print_exc()
                    env.primitive.mp_err = getattr(env.primitive, "mp_err", None) or "BaseMPException"
                    base_mp_trial += 1
                    nav_mp_success = False
                finally:
                    _restore_nav_debug_wrappers(nav_debug_originals)

                # Save timings to current_phase_logs
                nav_execution_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["base_sampling_time"][base_mp_trial] = getattr(env.primitive, "base_sampling_time", 0)
                phase_logs[env.execution_phase_ind]["base_mp_planning_time"][base_mp_trial] = getattr(env.primitive, "base_mp_planning_time", 0)
                phase_logs[env.execution_phase_ind]["base_mp_execution_time"][base_mp_trial] = round(nav_execution_finish_time - nav_execution_start_time, 2)

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for nav_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
                print(f"Visibility stats for nav_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"])

                if nav_mp_success and bool(int(os.environ.get("MOMAGEN_NAV_ACCEPTANCE_GUARD", "1") or 1)):
                    nav_guard_record = {
                        "phase": int(env.execution_phase_ind),
                        "base_mp_trial": int(base_mp_trial),
                        "enabled": True,
                        "reject": False,
                        "reason": None,
                        "ref_obj_name": getattr(ref_obj, "name", None),
                    }
                    reject_nav_trial = False

                    if ref_obj is not None and pre_nav_ref_obj_pose is not None:
                        try:
                            post_nav_ref_obj_pose = ref_obj.get_position_orientation()
                            pre_pos = _debug_to_np(pre_nav_ref_obj_pose[0])
                            post_pos = _debug_to_np(post_nav_ref_obj_pose[0])
                            pre_quat = _debug_to_np(pre_nav_ref_obj_pose[1])
                            post_quat = _debug_to_np(post_nav_ref_obj_pose[1])
                            obj_displacement = None
                            obj_xy_displacement = None
                            if pre_pos is not None and post_pos is not None:
                                obj_displacement = float(np.linalg.norm(post_pos - pre_pos))
                                obj_xy_displacement = float(np.linalg.norm(post_pos[:2] - pre_pos[:2]))
                            max_obj_displacement = float(
                                os.environ.get("MOMAGEN_NAV_MAX_REF_OBJ_DISPLACEMENT", "0.10") or 0.10
                            )
                            max_obj_xy_displacement = float(
                                os.environ.get(
                                    "MOMAGEN_NAV_MAX_REF_OBJ_XY_DISPLACEMENT",
                                    str(max_obj_displacement),
                                )
                                or max_obj_displacement
                            )
                            nav_guard_record.update(
                                {
                                    "pre_ref_obj_pos": None if pre_pos is None else pre_pos.tolist(),
                                    "post_ref_obj_pos": None if post_pos is None else post_pos.tolist(),
                                    "pre_ref_obj_quat": None if pre_quat is None else pre_quat.tolist(),
                                    "post_ref_obj_quat": None if post_quat is None else post_quat.tolist(),
                                    "ref_obj_displacement": obj_displacement,
                                    "ref_obj_xy_displacement": obj_xy_displacement,
                                    "max_ref_obj_displacement": max_obj_displacement,
                                    "max_ref_obj_xy_displacement": max_obj_xy_displacement,
                                }
                            )
                            if (
                                obj_displacement is not None
                                and obj_displacement > max_obj_displacement
                            ) or (
                                obj_xy_displacement is not None
                                and obj_xy_displacement > max_obj_xy_displacement
                            ):
                                reject_nav_trial = True
                                nav_guard_record["reason"] = "ref_obj_displaced"
                        except Exception as guard_e:
                            nav_guard_record["object_displacement_error"] = f"{type(guard_e).__name__}: {guard_e}"

                    require_final_visibility = (
                        enable_visibility_constraint
                        and bool(int(os.environ.get("MOMAGEN_NAV_REQUIRE_FINAL_VISIBILITY", "1") or 1))
                    )
                    if require_final_visibility and not reject_nav_trial:
                        visible_helper = getattr(env.primitive, "_object_visible_from_current_cameras", None)
                        if callable(visible_helper) and ref_obj is not None:
                            try:
                                final_visible = bool(visible_helper(ref_obj))
                            except Exception as visibility_e:
                                final_visible = False
                                nav_guard_record["final_visibility_error"] = (
                                    f"{type(visibility_e).__name__}: {visibility_e}"
                                )
                        else:
                            final_visible = not bool(
                                int(os.environ.get("MOMAGEN_NAV_FAIL_CLOSED_ON_MISSING_VISIBILITY_HELPER", "1") or 1)
                            )
                            nav_guard_record["final_visibility_error"] = "visibility_helper_unavailable"
                        nav_guard_record["final_visible"] = bool(final_visible)
                        if not final_visible:
                            reject_nav_trial = True
                            nav_guard_record["reason"] = "final_visibility_failed"

                    nav_guard_record["reject"] = bool(reject_nav_trial)
                    phase_logs[env.execution_phase_ind].setdefault("nav_acceptance_guard", []).append(nav_guard_record)
                    print("[MOMAGEN_NAV_ACCEPTANCE_GUARD] " + json.dumps(nav_guard_record, default=str), flush=True)

                    if reject_nav_trial:
                        env.primitive.mp_err = nav_guard_record["reason"] or "NavAcceptanceGuardFailed"
                        og.sim.load_state(init_state)
                        for _ in range(5):
                            og.sim.step()
                        env.global_env_step = init_global_env_step
                        self.reset_visibility_counter(env)
                        base_mp_trial += 1
                        nav_mp_success = False
                        continue

                if not nav_mp_success:
                    # This will happen if
                    # 1. base sampling fails
                    # 2. base MP fails.
                    # 3. base execution fails to converge

                    # In case #3, we actually step physics in OG, so we need to reset the state
                    if env.primitive.mp_err in [
                        "BaseExecutionBaseTargetNotReached",
                        "BaseExecutionArmTorsoTargetNotReached",
                    ] or num_phase_steps > 0:
                        og.sim.load_state(init_state)
                        for _ in range(5): og.sim.step()

                        # Reset the visibility stats
                        self.reset_visibility_counter(env)

                    continue

                env.err = env.primitive.mp_err
                MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
                left_mp_ranges = [init_global_env_step, env.global_env_step]
                right_mp_ranges = [init_global_env_step, env.global_env_step]
                results = dict(
                    states=states,
                    observations=observations,
                    datagen_infos=datagen_infos,
                    actions=np.array(actions),
                    success=bool(success["task"]),
                    mp_end_steps=MP_end_step_local_list,
                    subtask_lengths=local_env_step,
                    left_mp_ranges=left_mp_ranges,
                    right_mp_ranges=right_mp_ranges,
                    retry_nav=False,
                    observations_info=observations_info
                )
                # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase).
                # In this case MP succeeded and phase was actually executed
                env.execution_phase_ind += 1
                env.phases_completed_wo_mp_err += 1
                return results
        # ============================================== Base Navigation ==============================================

        if phase_type != "navigation":
            # =============================================== Arm MP Planning =============================================
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            init_global_env_step = env.global_env_step
            arm_mp_execution_start_time = time.time()
            # success = {k: False for k in env.is_success()} # success metrics

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            wholebody_cover_replay_enabled = bool(
                int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_COVER_REPLAY", "0") or 0)
            )
            wholebody_cover_replay_phase_in_range = bool(
                int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP", "0") or 0)
                and int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MIN_PHASE", "0") or 0)
                <= int(env.execution_phase_ind)
                <= int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MAX_PHASE", "999999") or 999999)
            )
            if wholebody_cover_replay_enabled and wholebody_cover_replay_phase_in_range:
                original_mp_end_steps = [int(step) for step in cur_subtask_end_step_MP]
                cur_subtask_end_step_MP = [len(seq), len(seq)]
                phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp_cover_replay", []).append(
                    {
                        "enabled": True,
                        "applied": True,
                        "phase": int(env.execution_phase_ind),
                        "original_mp_end_steps": original_mp_end_steps,
                        "new_mp_end_steps": [int(step) for step in cur_subtask_end_step_MP],
                        "reason": "wholebody_plan_through_contact_replay",
                    }
                )

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # # Temporary: This is just to capture the first image after navigating to the teacup, just for visualization
            # if object_ref["arm_left"] == "teacup" and grasp_init_views_video_writer is not None:
            #     robot_name = env.env.robots[0].name
            #     obs, obs_info = env.get_observation()
            #     ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
            #     # eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            #     # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            #     concatenated_img = hori_concatenate_image([ego_img])
            #     grasp_init_views_video_writer.append_data(concatenated_img)


            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

            # If one of the arms does not have a ref object in this subtask, keep its MP target
            # at the current EEF pose. Some source demos still contain waypoint rows for the
            # inactive arm; using those rows as constraints can make single-arm subtasks
            # spuriously fail IK on current R1Pro / CuRobo configs.
            if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")
            elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")


            # If at least one hand has motion planner waypoints, plan the motion
            if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                target_pos = {
                    robot.eef_link_names["left"]: left_waypoint_pos,
                    robot.eef_link_names["right"]: right_waypoint_pos,
                }
                target_quat = {
                    robot.eef_link_names["left"]: left_waypoint_ori,
                    robot.eef_link_names["right"]: right_waypoint_ori,
                }
                # If both hands have motion planner waypoints, we use the arm + torso embodiment
                # If only one of the hands has motion planner waypoints, we use the arm embodiment only because
                # when we replay the waypoints for the other hand, we assume the torso is fixed.
                has_arm_no_torso_emb_sel = hasattr(CuRoboEmbodimentSelection, "ARM_NO_TORSO")
                arm_no_torso_emb_sel = getattr(
                    CuRoboEmbodimentSelection,
                    "ARM_NO_TORSO",
                    CuRoboEmbodimentSelection.ARM,
                )
                emb_sel = CuRoboEmbodimentSelection.ARM if len(left_mp_waypoints) > 0 and len(right_mp_waypoints) > 0 else arm_no_torso_emb_sel

                # To test MP in arm_no_toso mode instead of arm mode, uncomment the line below
                emb_sel = arm_no_torso_emb_sel
                wholebody_arm_mp_enabled = bool(
                    int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP", "0") or 0)
                )
                wholebody_min_phase = int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MIN_PHASE", "0") or 0)
                wholebody_max_phase = int(
                    os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP_MAX_PHASE", "999999") or 999999
                )
                wholebody_phase_in_range = bool(
                    wholebody_min_phase <= int(env.execution_phase_ind) <= wholebody_max_phase
                )
                requested_main_ee_link = None
                main_ee_link_by_phase_raw = os.environ.get("MOMAGEN_ARM_MP_MAIN_EE_LINK_BY_PHASE", "").strip()
                main_ee_link_parse_errors = []
                if main_ee_link_by_phase_raw:
                    phase_ind = int(env.execution_phase_ind)
                    for raw_entry in main_ee_link_by_phase_raw.replace(";", ",").split(","):
                        entry = raw_entry.strip()
                        if not entry:
                            continue
                        if ":" not in entry:
                            main_ee_link_parse_errors.append({"entry": entry, "reason": "missing_colon"})
                            continue
                        phase_spec, link_name = [part.strip() for part in entry.split(":", 1)]
                        if not phase_spec or not link_name:
                            main_ee_link_parse_errors.append({"entry": entry, "reason": "empty_phase_or_link"})
                            continue
                        try:
                            if "-" in phase_spec:
                                phase_start_raw, phase_end_raw = [part.strip() for part in phase_spec.split("-", 1)]
                                phase_start = int(phase_start_raw)
                                phase_end = int(phase_end_raw)
                            else:
                                phase_start = phase_end = int(phase_spec)
                        except ValueError:
                            main_ee_link_parse_errors.append({"entry": entry, "reason": "invalid_phase"})
                            continue
                        if phase_start <= phase_ind <= phase_end:
                            requested_main_ee_link = link_name
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_main_ee_link_by_phase", []).append(
                        {
                            "enabled": True,
                            "phase": int(env.execution_phase_ind),
                            "raw": main_ee_link_by_phase_raw,
                            "selected_main_ee_link": requested_main_ee_link,
                            "parse_errors": main_ee_link_parse_errors,
                        }
                    )
                if wholebody_arm_mp_enabled and wholebody_phase_in_range:
                    wholebody_emb_sel = CuRoboEmbodimentSelection.DEFAULT
                    if requested_main_ee_link:
                        wholebody_emb_sel = default_embodiment_variant(requested_main_ee_link)
                    if wholebody_emb_sel in getattr(getattr(env, "cmg", None), "mg", {}):
                        emb_sel = wholebody_emb_sel
                    elif CuRoboEmbodimentSelection.DEFAULT in getattr(getattr(env, "cmg", None), "mg", {}):
                        emb_sel = CuRoboEmbodimentSelection.DEFAULT
                        if requested_main_ee_link:
                            phase_logs[env.execution_phase_ind].setdefault("arm_mp_main_ee_link_by_phase", []).append(
                                {
                                    "enabled": True,
                                    "phase": int(env.execution_phase_ind),
                                    "selected_main_ee_link": requested_main_ee_link,
                                    "applied": False,
                                    "reason": "variant_embodiment_unavailable",
                                    "requested_emb_sel": str(wholebody_emb_sel),
                                }
                            )
                    else:
                        wholebody_arm_mp_enabled = False
                        phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp", []).append(
                            {
                                "enabled": True,
                                "applied": False,
                                "reason": "default_embodiment_unavailable",
                                "phase": int(env.execution_phase_ind),
                            }
                        )
                elif wholebody_arm_mp_enabled:
                    wholebody_arm_mp_enabled = False
                phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp", []).append(
                    {
                        "enabled": bool(int(os.environ.get("MOMAGEN_WHOLEBODY_ARM_MP", "0") or 0)),
                        "applied": bool(wholebody_arm_mp_enabled),
                        "phase": int(env.execution_phase_ind),
                        "phase_in_range": bool(wholebody_phase_in_range),
                        "emb_sel": str(emb_sel),
                        "main_ee_link": str(getattr(env.cmg, "ee_link", {}).get(emb_sel, None)),
                        "additional_links": list(getattr(env.cmg, "additional_links", {}).get(emb_sel, [])),
                        "base_link": str(getattr(env.cmg, "base_link", {}).get(emb_sel, None)),
                    }
                )

                # # Option 1: Use template to know attached objects
                # if attached_obj is None:
                #     attached_obj_scale = None
                # else:
                #     attached_obj_new = {}
                #     attached_obj_scale = {}
                #     for arm, obj_name in attached_obj.items():
                #         if obj_name is not None:
                #             attached_obj_new[robot.eef_link_names[arm]] = env.env.scene.object_registry("name", obj_name).root_link
                #             attached_obj_scale[robot.eef_link_names[arm]] = 0.9
                #     attached_obj = attached_obj_new

                # Option 2: Use OG to know attached objects
                retval = self.obtain_attached_object(env, robot)
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]
                grasp_action = retval["grasp_action"]
                ignore_attached_obj_for_arm_mp = bool(
                    int(os.environ.get("MOMAGEN_IGNORE_ATTACHED_OBJ_FOR_ARM_MP", "0") or 0)
                )

                # Some BEHAVIOR-1K D0/D1 configs encode carry phases as a single
                # manipulation phase even though the source segment contains base motion.
                # When R1Pro is already holding an object, an arm-only MP target can be
                # outside the reachable workspace. Optionally replay the source base
                # prefix while holding current arm joints / grippers, then plan the arm
                # motion from the moved base pose. Disabled by default; set
                # MOMAGEN_SOURCE_BASE_PREAPPROACH_STEPS=-1 to replay the whole current
                # phase source segment, or a positive integer to cap the prefix length.
                if (
                    source_base_preapproach_steps != 0
                    and int(os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_MIN_PHASE", "0") or 0)
                    <= int(env.execution_phase_ind)
                    <= int(os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_MAX_PHASE", "999999") or 999999)
                    and src_curr_phase_actions is not None
                    and attached_obj
                ):
                    src_actions_np = np.asarray(src_curr_phase_actions)
                    if src_actions_np.ndim >= 2 and src_actions_np.shape[0] > 0:
                        num_preapproach_steps = src_actions_np.shape[0]
                        if source_base_preapproach_steps > 0:
                            num_preapproach_steps = min(num_preapproach_steps, source_base_preapproach_steps)
                        base_idx = robot.controller_action_idx.get("base", getattr(robot, "base_action_idx", None))
                        if base_idx is not None and num_preapproach_steps > 0:
                            init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                            init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                            init_global_env_step = env.global_env_step
                            preapproach_mode = str(
                                os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_MODE", "source_action") or "source_action"
                            )
                            print(
                                f"[MOMAGEN_SOURCE_BASE_PREAPPROACH] executing {num_preapproach_steps} "
                                f"step(s), mode={preapproach_mode}"
                            )

                            preapproach_actions_iter = None
                            if preapproach_mode in ("source_pose_q_to_action", "target_eef_local_q_to_action"):
                                selected_source_base_arm = (
                                    "right"
                                    if _arm_has_attached_payload("right") or object_ref.get("arm_right") is not None
                                    else "left"
                                )
                                source_pose_2d, source_base_idx = _source_base_pose_for_nav_arm(selected_source_base_arm)
                                source_snap_record = None
                                target_eef_local_record = None
                                if preapproach_mode == "target_eef_local_q_to_action":
                                    # Carry phases may start from a generated grasp / manipulation posture whose
                                    # hand-to-base offset differs substantially from the source demo.  Moving only to
                                    # the source base pose can therefore transport the held object to a pose that is
                                    # still outside the arm IK basin.  This diagnostic mode keeps the current active
                                    # hand local offset fixed and solves for the base pose that would put that hand at
                                    # the transformed phase MP target, while still using the source yaw when available.
                                    active_link_name = robot.eef_link_names.get(selected_source_base_arm)
                                    active_target_pos = target_pos.get(active_link_name) if active_link_name is not None else None
                                    if active_target_pos is not None:
                                        current_base_q = robot.get_joint_positions()[robot.base_control_idx].detach().clone()
                                        current_eef_pos = robot.get_eef_pose(selected_source_base_arm)[0].to(
                                            dtype=current_base_q.dtype, device=current_base_q.device
                                        )
                                        target_yaw = (
                                            source_pose_2d[2]
                                            if source_pose_2d is not None and bool(th.isfinite(source_pose_2d).all())
                                            else current_base_q[2]
                                        )
                                        cur_yaw = current_base_q[2]
                                        cur_cos, cur_sin = th.cos(cur_yaw), th.sin(cur_yaw)
                                        tgt_cos, tgt_sin = th.cos(target_yaw), th.sin(target_yaw)
                                        eef_world_delta = current_eef_pos[:2] - current_base_q[:2]
                                        # R(-cur_yaw) @ eef_world_delta
                                        eef_local_xy = th.stack(
                                            [
                                                cur_cos * eef_world_delta[0] + cur_sin * eef_world_delta[1],
                                                -cur_sin * eef_world_delta[0] + cur_cos * eef_world_delta[1],
                                            ]
                                        )
                                        # active_target_xy - R(target_yaw) @ eef_local_xy
                                        target_xy = th.as_tensor(
                                            active_target_pos[:2], dtype=current_base_q.dtype, device=current_base_q.device
                                        )
                                        base_xy = target_xy - th.stack(
                                            [
                                                tgt_cos * eef_local_xy[0] - tgt_sin * eef_local_xy[1],
                                                tgt_sin * eef_local_xy[0] + tgt_cos * eef_local_xy[1],
                                            ]
                                        )
                                        source_pose_2d = th.stack([base_xy[0], base_xy[1], target_yaw]).to(dtype=th.float32)
                                        target_eef_local_record = {
                                            "active_arm": selected_source_base_arm,
                                            "active_link_name": active_link_name,
                                            "active_target_pos": _debug_array_value(active_target_pos),
                                            "current_base_q": _debug_array_value(current_base_q),
                                            "current_eef_pos": _debug_array_value(current_eef_pos),
                                            "eef_local_xy": _debug_array_value(eef_local_xy),
                                            "raw_target_base_pose_2d": _debug_array_value(source_pose_2d),
                                        }
                                source_pose_2d, source_snap_record = _snap_pose_to_current_traversable_component(source_pose_2d)
                                preapproach_record = {
                                    "mode": preapproach_mode,
                                    "selected_source_base_arm": selected_source_base_arm,
                                    "source_base_idx": None if source_base_idx is None else int(source_base_idx),
                                    "source_pose_2d": _debug_array_value(source_pose_2d),
                                    "source_snap": source_snap_record,
                                    "target_eef_local": target_eef_local_record,
                                }
                                phase_logs[env.execution_phase_ind].setdefault("source_base_preapproach_debug", []).append(
                                    preapproach_record
                                )
                                print(
                                    "[MOMAGEN_SOURCE_BASE_PREAPPROACH] "
                                    + json.dumps(preapproach_record, default=str),
                                    flush=True,
                                )
                                if source_pose_2d is not None and bool(th.isfinite(source_pose_2d).all()):
                                    current_base_q = robot.get_joint_positions()[robot.base_control_idx].detach().clone()
                                    target_base_q = source_pose_2d.to(dtype=current_base_q.dtype, device=current_base_q.device)
                                    # Interpolate in world x/y and wrapped yaw. q_to_action will convert each intermediate
                                    # world-frame base target into the HolonomicBaseJointController's local-frame command.
                                    yaw_delta = wrap_angle(target_base_q[2] - current_base_q[2])
                                    q_targets = []
                                    for step_i in range(num_preapproach_steps):
                                        frac = float(step_i + 1) / float(num_preapproach_steps)
                                        q_target = robot.get_joint_positions().detach().clone()
                                        q_target[robot.base_control_idx[0]] = current_base_q[0] + frac * (
                                            target_base_q[0] - current_base_q[0]
                                        )
                                        q_target[robot.base_control_idx[1]] = current_base_q[1] + frac * (
                                            target_base_q[1] - current_base_q[1]
                                        )
                                        q_target[robot.base_control_idx[2]] = current_base_q[2] + frac * yaw_delta
                                        q_targets.append(q_target)
                                    preapproach_actions_iter = q_targets

                            if preapproach_actions_iter is None:
                                preapproach_actions_iter = src_actions_np[:num_preapproach_steps]

                            preapproach_debug_interval = int(
                                os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_DEBUG_INTERVAL", "25") or 25
                            )
                            for preapproach_i, src_action in enumerate(preapproach_actions_iter):
                                action = env.primitive._empty_action()
                                if preapproach_mode in ("source_pose_q_to_action", "target_eef_local_q_to_action") and isinstance(src_action, th.Tensor):
                                    q_action = _joint_trajectory_point_to_action(robot, src_action).to(
                                        dtype=action.dtype, device=action.device
                                    )
                                    src_base_action = q_action[base_idx]
                                else:
                                    src_base_action = th.as_tensor(
                                        src_action[base_idx], dtype=action.dtype, device=action.device
                                    )
                                action[base_idx] = src_base_action
                                action[robot.arm_action_idx["left"]] = init_left_arm_pos
                                action[robot.arm_action_idx["right"]] = init_right_arm_pos
                                action[robot.gripper_action_idx["left"]] = grasp_action["left"]
                                action[robot.gripper_action_idx["right"]] = grasp_action["right"]
                                pre_base_pose = robot.get_position_orientation()
                                state = env.get_state()["states"]
                                obs, obs_info = env.get_obs_IL()
                                datagen_info = env_interface.get_datagen_info(action=action)
                                env.step(action, video_writer)
                                if preapproach_debug_interval > 0 and (
                                    preapproach_i == 0
                                    or preapproach_i == num_preapproach_steps - 1
                                    or preapproach_i % preapproach_debug_interval == 0
                                ):
                                    post_base_pose = robot.get_position_orientation()
                                    phase_logs[env.execution_phase_ind].setdefault(
                                        "source_base_preapproach_trace", []
                                    ).append(
                                        {
                                            "step": int(preapproach_i),
                                            "mode": preapproach_mode,
                                            "base_action": _debug_array_value(src_base_action),
                                            "pre_base_pos": _debug_array_value(pre_base_pose[0]),
                                            "post_base_pos": _debug_array_value(post_base_pose[0]),
                                            "pre_base_quat": _debug_array_value(pre_base_pose[1]),
                                            "post_base_quat": _debug_array_value(post_base_pose[1]),
                                        }
                                    )
                                local_env_step += 1
                                env.global_env_step += 1
                                states.append(state)
                                actions.append(action)
                                observations.append(obs)
                                observations_info.append(json.dumps(obs_info))
                                datagen_infos.append(datagen_info)
                                cur_success_metrics = env.is_success()
                                for k in success:
                                    success[k] = success[k] or cur_success_metrics[k]
                                if ref_obj is not None:
                                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            phase_logs[env.execution_phase_ind].setdefault("source_base_preapproach_steps", {})[0] = int(
                                env.global_env_step - init_global_env_step
                            )

                            if bool(int(os.environ.get("MOMAGEN_SOURCE_BASE_PREAPPROACH_ONLY", "0") or 0)):
                                _log_manip_debug(
                                    stage="after_source_base_preapproach_only",
                                    target_pos_by_link=target_pos,
                                    attached_obj_by_link=attached_obj,
                                )
                                print(
                                    "[MOMAGEN_SOURCE_BASE_PREAPPROACH_ONLY] "
                                    f"skipping arm MP after source-base transport; success={bool(success['task'])}"
                                )
                                preapproach_actions = [
                                    a.detach().cpu().numpy() if isinstance(a, th.Tensor) else np.asarray(a)
                                    for a in actions
                                ]
                                results = dict(
                                    states=states,
                                    observations=observations,
                                    datagen_infos=datagen_infos,
                                    actions=np.array(preapproach_actions),
                                    success=bool(success["task"]),
                                    mp_end_steps=[local_env_step, local_env_step],
                                    subtask_lengths=local_env_step,
                                    left_mp_ranges=[init_global_env_step, env.global_env_step],
                                    right_mp_ranges=[init_global_env_step, env.global_env_step],
                                    retry_nav=False,
                                    observations_info=observations_info,
                                )
                                env.execution_phase_ind += 1
                                env.phases_completed_wo_mp_err += 1
                                return results

                planning_attached_obj = attached_obj
                planning_attached_obj_scale = attached_obj_scale
                if ignore_attached_obj_for_arm_mp and attached_obj:
                    print("[MOMAGEN_IGNORE_ATTACHED_OBJ_FOR_ARM_MP] planning arm MP without attached-object collision")
                    planning_attached_obj = None
                    planning_attached_obj_scale = None

                # Option 2: If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                if has_arm_no_torso_emb_sel:
                    if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                        del target_pos["right_eef_link"]
                        del target_quat["right_eef_link"]
                    elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                        del target_pos["left_eef_link"]
                        del target_quat["left_eef_link"]

                _maybe_apply_toggle_marker_target_correction(target_pos, target_quat)
                _maybe_apply_toggle_marker_joint_staging_targets(target_pos, target_quat, emb_sel)

                _log_manip_debug(
                    stage="after_source_base_preapproach",
                    target_pos_by_link=target_pos,
                    attached_obj_by_link=attached_obj,
                )

                # # Check object visibility at start-of-manip step
                # try:
                #     obs, obs_info = env.get_observation()
                #     seg_instance = obs[f"{env.robot_name}::{env.robot_name}:eyes:Camera:0::seg_instance"]
                #     seg_instance_info = obs_info[f"{env.robot_name}"][f"{env.robot_name}:eyes:Camera:0"]["seg_instance"]
                #     key_of_coffee_cup = next((key for key, value in seg_instance_info.items() if value == "coffee_cup"), None)
                #     if key_of_coffee_cup is None:
                #         count = 0
                #     else:
                #         count = (seg_instance == key_of_coffee_cup).sum().item()
                #     if count > 150:
                #         env.obj_visible_at_start_of_manip = True
                # except Exception as e:


                # This is for retract behavior. We are not using this as of now, but let it be
                initial_left_eef_pose = robot.get_eef_pose("left")
                initial_right_eef_pose = robot.get_eef_pose("right")

                print("ARM MP START")
                eyes_target_pos, eyes_target_quat = None, None
                # NOTE: Keep this commented out. We won't be using soft visibility constraint with manipulation for now. As we are using ARM_NO_TORSO mode
                # if env.soft_visibility_constraint:
                #     obj_pose = ref_obj.get_position_orientation()
                #     eyes_target_pos = obj_pose[0]
                #     eyes_target_quat = obj_pose[1]

                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                    env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)


                arm_mp_diag_enabled = bool(int(os.environ.get("MOMAGEN_ARM_MP_DIAG", "0") or 0))
                arm_mp_diag_max_variants = int(os.environ.get("MOMAGEN_ARM_MP_DIAG_MAX_VARIANTS", "9") or 9)
                arm_mp_self_collision_check = bool(int(os.environ.get("MOMAGEN_ARM_MP_SELF_COLLISION_CHECK", "1") or 1))
                arm_mp_self_collision_disable_when_attached = bool(
                    int(os.environ.get("MOMAGEN_ARM_MP_SELF_COLLISION_DISABLE_WHEN_ATTACHED", "0") or 0)
                )
                disable_default_emb_sel_fallback = bool(
                    int(os.environ.get("MOMAGEN_DISABLE_DEFAULT_EMB_SEL_FALLBACK", "0") or 0)
                )
                arm_mp_diag_done = False

                def _run_arm_mp_failure_diagnostics(failed_status_value, failed_trial):
                    """Run a small set of diagnostic-only CuRobo probes after ARM TrajOpt failure."""
                    if not arm_mp_diag_enabled:
                        return None

                    diag_batch_size = env.primitive._motion_generator.batch_size
                    base_diag_target_pos = {
                        k: th.stack([v for _ in range(diag_batch_size)]) for k, v in target_pos.items()
                    }
                    base_diag_target_quat = {
                        k: th.stack([v for _ in range(diag_batch_size)]) for k, v in target_quat.items()
                    }
                    current_joint_pos = robot.get_joint_positions()
                    current_q_batch = th.stack([current_joint_pos for _ in range(diag_batch_size)])

                    diag_record = {
                        "trial": int(failed_trial),
                        "failed_status": failed_status_value,
                        "emb_sel": str(emb_sel),
                        "target_pos_by_link": {k: _debug_array_value(v) for k, v in target_pos.items()},
                        "target_quat_by_link": {k: _debug_array_value(v) for k, v in target_quat.items()},
                        "current_joint_pos": _debug_array_value(current_joint_pos),
                        "current_eef_pose": {
                            arm: _debug_pose_value(robot.get_eef_pose(arm)) for arm in ("left", "right")
                        },
                        "attached_obj_keys": list((attached_obj or {}).keys()) if attached_obj else [],
                        "planning_attached_obj_keys": list((planning_attached_obj or {}).keys()) if planning_attached_obj else [],
                        "variants": [],
                    }

                    def _diag_compute_variant(name, **variant_kwargs):
                        try:
                            diag_result = _compute_trajectories_with_paths(
                                env.cmg,
                                target_pos=copy.deepcopy(base_diag_target_pos),
                                target_quat=copy.deepcopy(base_diag_target_quat),
                                is_local=False,
                                max_attempts=int(os.environ.get("MOMAGEN_ARM_MP_DIAG_MAX_ATTEMPTS", "20") or 20),
                                timeout=float(os.environ.get("MOMAGEN_ARM_MP_DIAG_TIMEOUT", "30.0") or 30.0),
                                ik_fail_return=int(os.environ.get("MOMAGEN_ARM_MP_DIAG_IK_FAIL_RETURN", "10") or 10),
                                enable_finetune_trajopt=bool(
                                    int(os.environ.get("MOMAGEN_ARM_MP_DIAG_FINETUNE", "0") or 0)
                                ),
                                finetune_attempts=0,
                                return_full_result=True,
                                success_ratio=1.0 / diag_batch_size,
                                emb_sel=emb_sel,
                                **variant_kwargs,
                            )
                            diag_mp_results = diag_result[0]
                            diag_successes = getattr(diag_mp_results[0], "success", None) if diag_mp_results else None
                            diag_status_obj = getattr(diag_mp_results[0], "status", None) if diag_mp_results else None
                            diag_status_value = getattr(diag_status_obj, "value", str(diag_status_obj))
                            successes_np = _debug_to_np(diag_successes)
                            diag_record["variants"].append(
                                {
                                    "name": name,
                                    "status": diag_status_value,
                                    "success": bool(
                                        successes_np is not None and np.asarray(successes_np).astype(bool).any()
                                    ),
                                    "successes": None
                                    if successes_np is None
                                    else np.asarray(successes_np).astype(bool).tolist(),
                                    "ik_only": bool(variant_kwargs.get("ik_only", False)),
                                    "ik_world_collision_check": variant_kwargs.get("ik_world_collision_check", None),
                                    "attached_obj_keys": list((variant_kwargs.get("attached_obj") or {}).keys())
                                    if variant_kwargs.get("attached_obj")
                                    else [],
                                    "attached_obj_options": variant_kwargs.get("attached_obj_options"),
                                    "skip_obstacle_update": bool(variant_kwargs.get("skip_obstacle_update", False)),
                                }
                            )
                        except Exception as e:
                            diag_record["variants"].append(
                                {"name": name, "error": f"{type(e).__name__}: {e}"}
                            )

                    def _diag_collision_variant(name, **collision_kwargs):
                        try:
                            collision_result = env.cmg.check_collisions(
                                current_q_batch,
                                initial_joint_pos=current_joint_pos,
                                **collision_kwargs,
                            )
                            collision_np = _debug_to_np(collision_result)
                            diag_record["variants"].append(
                                {
                                    "name": name,
                                    "collision": None
                                    if collision_np is None
                                    else np.asarray(collision_np).astype(bool).tolist(),
                                    "any_collision": bool(
                                        collision_np is not None and np.asarray(collision_np).astype(bool).any()
                                    ),
                                    "attached_obj_keys": list((collision_kwargs.get("attached_obj") or {}).keys())
                                    if collision_kwargs.get("attached_obj")
                                    else [],
                                    "attached_obj_options": collision_kwargs.get("attached_obj_options"),
                                    "self_collision_check": bool(collision_kwargs.get("self_collision_check", True)),
                                }
                            )
                        except Exception as e:
                            diag_record["variants"].append(
                                {"name": name, "error": f"{type(e).__name__}: {e}"}
                            )

                    variants_run = 0
                    for name, kwargs in (
                        (
                            "ik_only_attached_world_collision",
                            dict(
                                attached_obj=planning_attached_obj,
                                attached_obj_scale=planning_attached_obj_scale,
                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                                skip_obstacle_update=True,
                                ik_only=True,
                                ik_world_collision_check=True,
                            ),
                        ),
                        (
                            "ik_only_attached_no_world_collision",
                            dict(
                                attached_obj=planning_attached_obj,
                                attached_obj_scale=planning_attached_obj_scale,
                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                                skip_obstacle_update=True,
                                ik_only=True,
                                ik_world_collision_check=False,
                            ),
                        ),
                        (
                            "ik_only_no_attached_world_collision",
                            dict(
                                attached_obj=None,
                                attached_obj_scale=None,
                                skip_obstacle_update=True,
                                ik_only=True,
                                ik_world_collision_check=True,
                            ),
                        ),
                        (
                            "trajopt_attached_no_self_collision",
                            dict(
                                attached_obj=planning_attached_obj,
                                attached_obj_scale=planning_attached_obj_scale,
                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                                skip_obstacle_update=True,
                                ik_only=False,
                                self_collision_check=False,
                            ),
                        ),
                        (
                            "trajopt_no_attached_skip_obstacles",
                            dict(
                                attached_obj=None,
                                attached_obj_scale=None,
                                skip_obstacle_update=True,
                                ik_only=False,
                            ),
                        ),
                    ):
                        if variants_run >= arm_mp_diag_max_variants:
                            break
                        _diag_compute_variant(name, **kwargs)
                        variants_run += 1

                    for name, kwargs in (
                        (
                            "current_q_collision_attached",
                            dict(
                                self_collision_check=True,
                                skip_obstacle_update=True,
                                attached_obj=planning_attached_obj,
                                attached_obj_scale=planning_attached_obj_scale,
                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                            ),
                        ),
                        (
                            "current_q_world_collision_attached_no_self",
                            dict(
                                self_collision_check=False,
                                skip_obstacle_update=True,
                                attached_obj=planning_attached_obj,
                                attached_obj_scale=planning_attached_obj_scale,
                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                            ),
                        ),
                        (
                            "current_q_collision_no_attached",
                            dict(
                                self_collision_check=True,
                                skip_obstacle_update=True,
                                attached_obj=None,
                                attached_obj_scale=None,
                            ),
                        ),
                        (
                            "current_q_world_collision_no_attached_no_self",
                            dict(
                                self_collision_check=False,
                                skip_obstacle_update=True,
                                attached_obj=None,
                                attached_obj_scale=None,
                            ),
                        ),
                    ):
                        if variants_run >= arm_mp_diag_max_variants:
                            break
                        _diag_collision_variant(name, **kwargs)
                        variants_run += 1

                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_failure_diag", []).append(diag_record)
                    print("[MOMAGEN_ARM_MP_DIAG] " + json.dumps(diag_record, default=str), flush=True)
                    return diag_record

                arm_mp_target_pos = target_pos
                arm_mp_target_quat = target_quat
                arm_mp_primary_link_override = None
                explicit_primary_link_override = os.environ.get("MOMAGEN_ARM_MP_PRIMARY_LINK_OVERRIDE", "")
                primary_link_by_phase_raw = os.environ.get("MOMAGEN_ARM_MP_PRIMARY_LINK_BY_PHASE", "").strip()
                if primary_link_by_phase_raw:
                    phase_primary_link_override = None
                    phase_primary_parse_errors = []
                    phase_ind = int(env.execution_phase_ind)
                    for raw_entry in primary_link_by_phase_raw.replace(";", ",").split(","):
                        entry = raw_entry.strip()
                        if not entry:
                            continue
                        if ":" not in entry:
                            phase_primary_parse_errors.append({"entry": entry, "reason": "missing_colon"})
                            continue
                        phase_spec, link_name = [part.strip() for part in entry.split(":", 1)]
                        if not phase_spec or not link_name:
                            phase_primary_parse_errors.append({"entry": entry, "reason": "empty_phase_or_link"})
                            continue
                        try:
                            if "-" in phase_spec:
                                phase_start_raw, phase_end_raw = [part.strip() for part in phase_spec.split("-", 1)]
                                phase_start = int(phase_start_raw)
                                phase_end = int(phase_end_raw)
                            else:
                                phase_start = phase_end = int(phase_spec)
                        except ValueError:
                            phase_primary_parse_errors.append({"entry": entry, "reason": "invalid_phase"})
                            continue
                        if phase_start <= phase_ind <= phase_end:
                            phase_primary_link_override = link_name
                    phase_primary_record = {
                        "enabled": True,
                        "phase": phase_ind,
                        "raw": primary_link_by_phase_raw,
                        "selected_primary_link": phase_primary_link_override,
                        "parse_errors": phase_primary_parse_errors,
                    }
                    if phase_primary_link_override is not None:
                        explicit_primary_link_override = phase_primary_link_override
                        phase_primary_record["applied_to_explicit_override"] = True
                    else:
                        phase_primary_record["applied_to_explicit_override"] = False
                        phase_primary_record["reason"] = "no_matching_phase"
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_primary_link_by_phase", []).append(
                        phase_primary_record
                    )
                    print(
                        "[MOMAGEN_ARM_MP_PRIMARY_LINK_BY_PHASE] "
                        + json.dumps(phase_primary_record, default=str),
                        flush=True,
                    )
                if explicit_primary_link_override:
                    explicit_primary_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_PRIMARY_LINK_MIN_PHASE", "0") or 0)
                    explicit_primary_max_phase = int(
                        os.environ.get("MOMAGEN_ARM_MP_PRIMARY_LINK_MAX_PHASE", "999999") or 999999
                    )
                    explicit_primary_phase_in_range = (
                        explicit_primary_min_phase <= int(env.execution_phase_ind) <= explicit_primary_max_phase
                    )
                    explicit_primary_record = {
                        "enabled": True,
                        "phase": int(env.execution_phase_ind),
                        "phase_in_range": bool(explicit_primary_phase_in_range),
                        "requested_primary_link": explicit_primary_link_override,
                        "target_keys": list(arm_mp_target_pos.keys()),
                    }
                    if explicit_primary_phase_in_range and explicit_primary_link_override in arm_mp_target_pos:
                        arm_mp_primary_link_override = explicit_primary_link_override
                        explicit_primary_record.update(
                            {
                                "applied": True,
                                "primary_link_override": arm_mp_primary_link_override,
                            }
                        )
                    else:
                        explicit_primary_record.update(
                            {
                                "applied": False,
                                "reason": "phase_out_of_range"
                                if not explicit_primary_phase_in_range
                                else "requested_link_missing_from_targets",
                            }
                        )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_primary_link_override", []).append(
                        explicit_primary_record
                    )
                    print(
                        "[MOMAGEN_ARM_MP_PRIMARY_LINK_OVERRIDE] "
                        + json.dumps(explicit_primary_record, default=str),
                        flush=True,
                    )
                filter_uncoordinated_arm_mp_targets = bool(
                    int(os.environ.get("MOMAGEN_FILTER_UNCOORDINATED_ARM_MP_TARGETS", "0") or 0)
                )
                if filter_uncoordinated_arm_mp_targets and phase_type == "uncoordinated":
                    filter_min_phase = int(os.environ.get("MOMAGEN_FILTER_UNCOORDINATED_ARM_MP_MIN_PHASE", "0") or 0)
                    filter_max_phase = int(
                        os.environ.get("MOMAGEN_FILTER_UNCOORDINATED_ARM_MP_MAX_PHASE", "999999") or 999999
                    )
                    active_arms = []
                    for arm_name in ("left", "right"):
                        if object_ref.get(f"arm_{arm_name}") is not None or attached_obj.get(
                            robot.eef_link_names[arm_name]
                        ) is not None:
                            active_arms.append(arm_name)
                    filter_record = {
                        "enabled": True,
                        "phase": int(env.execution_phase_ind),
                        "phase_type": phase_type,
                        "phase_in_range": bool(filter_min_phase <= int(env.execution_phase_ind) <= filter_max_phase),
                        "object_ref": {k: str(v) for k, v in object_ref.items()},
                        "attached_obj_keys": list((attached_obj or {}).keys()) if attached_obj else [],
                        "active_arms": active_arms,
                        "raw_target_keys": list(target_pos.keys()),
                        "emb_sel": str(emb_sel),
                        "curobo_main_ee_link": str(getattr(env.cmg, "ee_link", {}).get(emb_sel, None)),
                        "curobo_additional_links": list(getattr(env.cmg, "additional_links", {}).get(emb_sel, [])),
                    }
                    if filter_record["phase_in_range"] and len(active_arms) == 1:
                        active_link_name = robot.eef_link_names[active_arms[0]]
                        if active_link_name in target_pos and active_link_name in target_quat:
                            arm_mp_target_pos = {active_link_name: target_pos[active_link_name]}
                            arm_mp_target_quat = {active_link_name: target_quat[active_link_name]}
                            arm_mp_primary_link_override = active_link_name
                            filter_record["applied"] = True
                            filter_record["filtered_target_keys"] = list(arm_mp_target_pos.keys())
                            filter_record["primary_link_override"] = arm_mp_primary_link_override
                        else:
                            filter_record["applied"] = False
                            filter_record["reason"] = "active_link_missing_from_targets"
                    else:
                        filter_record["applied"] = False
                        filter_record["reason"] = (
                            "phase_out_of_range" if not filter_record["phase_in_range"] else "active_arm_count_not_one"
                        )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_target_filter", []).append(filter_record)
                    print("[MOMAGEN_ARM_MP_TARGET_FILTER] " + json.dumps(filter_record, default=str), flush=True)

                active_arms_override_raw = os.environ.get("MOMAGEN_ARM_MP_ACTIVE_ARMS_OVERRIDE", "").strip()
                if active_arms_override_raw:
                    override_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_ACTIVE_ARMS_MIN_PHASE", "0") or 0)
                    override_max_phase = int(
                        os.environ.get("MOMAGEN_ARM_MP_ACTIVE_ARMS_MAX_PHASE", "999999") or 999999
                    )
                    override_phase_in_range = override_min_phase <= int(env.execution_phase_ind) <= override_max_phase
                    requested_arms = [
                        arm.strip().lower()
                        for arm in active_arms_override_raw.split(",")
                        if arm.strip()
                    ]
                    invalid_arms = [arm for arm in requested_arms if arm not in ("left", "right")]
                    override_record = {
                        "enabled": True,
                        "phase": int(env.execution_phase_ind),
                        "phase_type": phase_type,
                        "phase_in_range": bool(override_phase_in_range),
                        "requested_active_arms": requested_arms,
                        "invalid_active_arms": invalid_arms,
                        "target_keys_before": list(arm_mp_target_pos.keys()),
                        "primary_link_before": arm_mp_primary_link_override,
                    }
                    if invalid_arms:
                        raise ValueError(
                            "MOMAGEN_ARM_MP_ACTIVE_ARMS_OVERRIDE only supports comma-separated left/right values"
                        )
                    if override_phase_in_range and requested_arms:
                        requested_links = [
                            robot.eef_link_names[arm]
                            for arm in requested_arms
                            if robot.eef_link_names[arm] in arm_mp_target_pos
                            and robot.eef_link_names[arm] in arm_mp_target_quat
                        ]
                        if requested_links:
                            arm_mp_target_pos = {
                                link_name: arm_mp_target_pos[link_name] for link_name in requested_links
                            }
                            arm_mp_target_quat = {
                                link_name: arm_mp_target_quat[link_name] for link_name in requested_links
                            }
                            arm_mp_primary_link_override = requested_links[0]
                            override_record.update(
                                {
                                    "applied": True,
                                    "target_keys_after": list(arm_mp_target_pos.keys()),
                                    "primary_link_override": arm_mp_primary_link_override,
                                }
                            )
                        else:
                            override_record.update(
                                {
                                    "applied": False,
                                    "reason": "requested_links_missing_from_targets",
                                }
                            )
                    else:
                        override_record.update(
                            {
                                "applied": False,
                                "reason": "phase_out_of_range"
                                if not override_phase_in_range
                                else "no_requested_arms",
                            }
                        )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_active_arms_override", []).append(
                        override_record
                    )
                    print(
                        "[MOMAGEN_ARM_MP_ACTIVE_ARMS_OVERRIDE] "
                        + json.dumps(override_record, default=str),
                        flush=True,
                    )

                hold_base_as_primary = bool(int(os.environ.get("MOMAGEN_ARM_MP_HOLD_BASE_AS_PRIMARY", "0") or 0))
                if hold_base_as_primary and is_default_embodiment(emb_sel):
                    hold_base_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_HOLD_BASE_MIN_PHASE", "0") or 0)
                    hold_base_max_phase = int(
                        os.environ.get("MOMAGEN_ARM_MP_HOLD_BASE_MAX_PHASE", "999999") or 999999
                    )
                    base_link_name = str(getattr(env.cmg, "base_link", {}).get(emb_sel, ""))
                    phase_in_range = hold_base_min_phase <= int(env.execution_phase_ind) <= hold_base_max_phase
                    hold_base_record = {
                        "enabled": True,
                        "phase": int(env.execution_phase_ind),
                        "phase_in_range": bool(phase_in_range),
                        "emb_sel": str(emb_sel),
                        "base_link": base_link_name,
                        "previous_primary_link_override": arm_mp_primary_link_override,
                        "target_keys_before": list(arm_mp_target_pos.keys()),
                    }
                    if phase_in_range and base_link_name in getattr(robot, "links", {}):
                        base_pos, base_quat = robot.links[base_link_name].get_position_orientation()
                        arm_mp_target_pos = dict(arm_mp_target_pos)
                        arm_mp_target_quat = dict(arm_mp_target_quat)
                        arm_mp_target_pos[base_link_name] = th.as_tensor(base_pos, dtype=th.float32)
                        arm_mp_target_quat[base_link_name] = th.as_tensor(base_quat, dtype=th.float32)
                        arm_mp_primary_link_override = base_link_name
                        hold_base_record.update(
                            {
                                "applied": True,
                                "base_target_pos": _debug_array_value(arm_mp_target_pos[base_link_name]),
                                "base_target_quat": _debug_array_value(arm_mp_target_quat[base_link_name]),
                                "target_keys_after": list(arm_mp_target_pos.keys()),
                                "primary_link_override": arm_mp_primary_link_override,
                            }
                        )
                    else:
                        hold_base_record.update(
                            {
                                "applied": False,
                                "reason": "phase_out_of_range" if not phase_in_range else "base_link_missing",
                            }
                        )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_hold_base_primary", []).append(
                        hold_base_record
                    )
                    print("[MOMAGEN_ARM_MP_HOLD_BASE_PRIMARY] " + json.dumps(hold_base_record, default=str), flush=True)

                # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                num_tries = 3
                arm_mp_trial = 0
                default_emb_sel_fallback_used = False
                attempted_emb_sels = {str(emb_sel)}
                new_target_pos = copy.deepcopy(arm_mp_target_pos)
                while True:

                    # Base condition
                    if arm_mp_trial > 0:
                        status_value = _mp_status_value(mp_results[0])
                        # # Trying a hacky way to reduce the IK failure. Basically moving the robot base a bit towards the object.
                        # # This does not ensure collision-free motion
                        # if "IK Fail" in mp_results[0].status.value:
                        #     obj_pos = ref_obj.get_position_orientation()[0][:2]
                        #     robot_base_pose = env.robot.get_position_orientation()
                        #     robot_base_pos = robot_base_pose[0][:2]
                        #     vec = obj_pos - robot_base_pos
                        #     vec = vec / np.linalg.norm(vec)
                        #     for _ in range(10):
                        #         joint_pos = env.robot.get_joint_positions()
                        #         joint_pos[:2] = joint_pos[:2] + (vec * 0.01)
                        #         action = env.robot.q_to_action(joint_pos).cpu().numpy()
                        #         # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        #         if left_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                        #         if right_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]

                        #         state = env.get_state()["states"]
                        #         obs, obs_info = env.get_obs_IL()
                        #         datagen_info = env_interface.get_datagen_info(action=action)
                        #         env.step(action, video_writer)
                        #         local_env_step += 1
                        #         env.global_env_step += 1
                        #         states.append(state)
                        #         actions.append(action)
                        #         observations.append(obs)
                        #         datagen_infos.append(datagen_info)

                        if ("IK Fail" in status_value or "TrajOpt Fail" in status_value) and env.retry_nav_on_arm_mp_failure:
                            results = dict(
                                states=states,
                                observations=observations,
                                datagen_infos=datagen_infos,
                                actions=np.array(actions),
                                success=bool(success["task"]),
                                retry_nav=True,
                                observations_info=observations_info
                            )
                            return results

                        # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                        if env.retry_nav_on_arm_mp_failure:
                            base_condition = arm_mp_trial == num_tries
                        else:
                            base_condition = arm_mp_trial == num_tries or ("IK Fail" in status_value)

                        if base_condition:
                            print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                            if "TrajOpt Fail" in status_value:
                                env.err = "ArmMPTrajOptFailed"
                            elif "IK Fail" in status_value:
                                env.err = "ArmMPIKFailed"
                            else:
                                env.err = "ArmMPOtherFailed"
                            env.valid_env = False
                            env.execution_phase_ind += 1
                            return None

                    # Aggregate target_pos and target_quat to match batch_size
                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                    new_target_quat = {
                        k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)])
                        for k, v in arm_mp_target_quat.items()
                    }

                    arm_mp_planning_start_time = time.time()
                    effective_arm_mp_self_collision_check = arm_mp_self_collision_check
                    if arm_mp_self_collision_disable_when_attached and planning_attached_obj:
                        effective_arm_mp_self_collision_check = False
                    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                    old_primary_link_override = os.environ.get("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE")
                    if arm_mp_primary_link_override is not None:
                        os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = arm_mp_primary_link_override
                    restore_joint_limits = None
                    clamp_base_limits_record = None
                    clamp_base_limits = bool(
                        int(os.environ.get("MOMAGEN_ARM_MP_CLAMP_BASE_JOINT_LIMITS", "0") or 0)
                    )
                    if clamp_base_limits and is_default_embodiment(emb_sel):
                        clamp_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_CLAMP_BASE_MIN_PHASE", "0") or 0)
                        clamp_max_phase = int(
                            os.environ.get("MOMAGEN_ARM_MP_CLAMP_BASE_MAX_PHASE", "999999") or 999999
                        )
                        clamp_phase_in_range = clamp_min_phase <= int(env.execution_phase_ind) <= clamp_max_phase
                        max_xy = float(os.environ.get("MOMAGEN_ARM_MP_CLAMP_BASE_MAX_XY", "0.35") or 0.35)
                        max_yaw = float(
                            os.environ.get("MOMAGEN_ARM_MP_CLAMP_BASE_MAX_YAW", "1.57079632679") or 1.57079632679
                        )
                        clamp_base_limits_record = {
                            "enabled": True,
                            "phase": int(env.execution_phase_ind),
                            "phase_in_range": bool(clamp_phase_in_range),
                            "emb_sel": str(emb_sel),
                            "max_xy": max_xy,
                            "max_yaw": max_yaw,
                        }
                        if clamp_phase_in_range and hasattr(robot, "base_control_idx"):
                            try:
                                current_q_for_limits = robot.get_joint_positions().detach().cpu()
                                base_control_idx = _safe_index_tensor(robot.base_control_idx)
                                base_q = current_q_for_limits[base_control_idx]
                                overrides = {
                                    "base_footprint_x_joint": (float(base_q[0] - max_xy), float(base_q[0] + max_xy)),
                                    "base_footprint_y_joint": (float(base_q[1] - max_xy), float(base_q[1] + max_xy)),
                                    "base_footprint_rz_joint": (
                                        float(base_q[2] - max_yaw),
                                        float(base_q[2] + max_yaw),
                                    ),
                                }
                                restore_joint_limits = _temporarily_clamp_curobo_joint_limits(
                                    env.cmg, emb_sel, overrides
                                )
                                clamp_base_limits_record.update(
                                    {
                                        "applied": restore_joint_limits is not None,
                                        "num_joint_limit_sets": None
                                        if restore_joint_limits is None
                                        else getattr(restore_joint_limits, "num_joint_limit_sets", None),
                                        "base_q": _debug_to_np(base_q).tolist(),
                                        "joint_limit_overrides": {
                                            k: [float(v[0]), float(v[1])] for k, v in overrides.items()
                                        },
                                    }
                                )
                                if restore_joint_limits is None:
                                    clamp_base_limits_record["reason"] = "no_matching_curobo_joint_limits"
                            except Exception as exc:
                                clamp_base_limits_record.update({"applied": False, "error": str(exc)})
                        else:
                            clamp_base_limits_record.update(
                                {
                                    "applied": False,
                                    "reason": "phase_out_of_range"
                                    if not clamp_phase_in_range
                                    else "robot_missing_base_control_idx",
                                }
                            )
                        phase_logs[env.execution_phase_ind].setdefault("arm_mp_clamp_base_joint_limits", []).append(
                            clamp_base_limits_record
                        )
                        print(
                            "[MOMAGEN_ARM_MP_CLAMP_BASE_JOINT_LIMITS] "
                            + json.dumps(clamp_base_limits_record, default=str),
                            flush=True,
                        )
                    try:
                        mp_results, traj_paths = _compute_trajectories_with_paths(env.cmg,
                            target_pos=new_target_pos,
                            target_quat=new_target_quat,
                            is_local=False,
                            max_attempts=50,
                            timeout=60.0,
                            ik_fail_return=10,
                            enable_finetune_trajopt=True,
                            finetune_attempts=1,
                            return_full_result=True,
                            success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                            attached_obj=planning_attached_obj,
                            attached_obj_scale=planning_attached_obj_scale,
                            attached_obj_options=_attached_payload_options(planning_attached_obj),
                            self_collision_check=effective_arm_mp_self_collision_check,
                            emb_sel=emb_sel,
                        )
                    finally:
                        if restore_joint_limits is not None:
                            restore_joint_limits()
                        if arm_mp_primary_link_override is not None:
                            if old_primary_link_override is None:
                                os.environ.pop("OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE", None)
                            else:
                                os.environ["OMNIGIBSON_CUROBO_PRIMARY_LINK_OVERRIDE"] = old_primary_link_override
                    arm_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                    successes = mp_results[0].success
                    print("Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()

                    status_value = _mp_status_value(mp_results[0])
                    successes_np = _debug_to_np(successes)
                    arm_mp_status_record = {
                        "trial": int(arm_mp_trial),
                        "status": status_value,
                        "success": bool(len(success_idx) > 0),
                        "success_idx": success_idx.tolist(),
                        "successes": None if successes_np is None else successes_np.astype(bool).tolist(),
                        "emb_sel": str(emb_sel),
                        "attached_obj_keys": list((planning_attached_obj or {}).keys()) if planning_attached_obj else [],
                        "attached_obj_options": _attached_payload_options(planning_attached_obj),
                        "ignored_attached_obj_for_arm_mp": bool(ignore_attached_obj_for_arm_mp and attached_obj),
                        "self_collision_check": bool(effective_arm_mp_self_collision_check),
                        "self_collision_check_base_setting": bool(arm_mp_self_collision_check),
                        "self_collision_disable_when_attached": bool(arm_mp_self_collision_disable_when_attached),
                        "primary_link_override": arm_mp_primary_link_override,
                    }
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_status", {})[
                        int(arm_mp_trial)
                    ] = arm_mp_status_record
                    print("[MOMAGEN_ARM_MP_STATUS] " + json.dumps(arm_mp_status_record, default=str), flush=True)

                    planned_fk_failed_links_for_retry = []
                    planned_fk_admission_enabled = bool(
                        int(os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_PLANNED_FK_ADMISSION", "1") or 1)
                    )
                    planned_fk_admission_min_phase = int(
                        os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_PLANNED_FK_ADMISSION_MIN_PHASE", "0") or 0
                    )
                    planned_fk_admission_max_phase = int(
                        os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_PLANNED_FK_ADMISSION_MAX_PHASE", "999999")
                        or 999999
                    )
                    planned_fk_admission_max_pos_err = float(
                        os.environ.get(
                            "MOMAGEN_COORDINATED_MULTI_EE_PLANNED_FK_ADMISSION_MAX_POS_ERR",
                            os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION_MAX_POS_ERR", "0.05"),
                        )
                        or 0.05
                    )
                    planned_fk_admission_phase_in_range = (
                        planned_fk_admission_min_phase
                        <= int(env.execution_phase_ind)
                        <= planned_fk_admission_max_phase
                    )
                    planned_fk_admission_target_links = [
                        link_name
                        for link_name in robot.eef_link_names.values()
                        if link_name in arm_mp_target_pos and link_name in arm_mp_target_quat
                    ]
                    planned_fk_admission_applies = bool(
                        planned_fk_admission_enabled
                        and planned_fk_admission_phase_in_range
                        and phase_type == "coordinated"
                        and len(planned_fk_admission_target_links) >= 2
                        and is_default_embodiment(emb_sel)
                        and len(success_idx) > 0
                    )
                    if planned_fk_admission_enabled:
                        planned_fk_admission_record = {
                            "enabled": True,
                            "applied": bool(planned_fk_admission_applies),
                            "phase": int(env.execution_phase_ind),
                            "phase_type": phase_type,
                            "emb_sel": str(emb_sel),
                            "trial": int(arm_mp_trial),
                            "phase_in_range": bool(planned_fk_admission_phase_in_range),
                            "min_phase": planned_fk_admission_min_phase,
                            "max_phase": planned_fk_admission_max_phase,
                            "max_pos_err": planned_fk_admission_max_pos_err,
                            "target_links": planned_fk_admission_target_links,
                            "raw_success_idx": success_idx.tolist(),
                            "accepted_success_idx": success_idx.tolist(),
                            "candidates": [],
                        }
                        planned_payload_collision_validation_enabled = bool(
                            int(
                                os.environ.get(
                                    "MOMAGEN_COORDINATED_ATTACHED_PAYLOAD_PAIR_COLLISION_VALIDATION",
                                    "1",
                                )
                                or 1
                            )
                        )
                        planned_payload_collision_margin = float(
                            os.environ.get(
                                "MOMAGEN_COORDINATED_ATTACHED_PAYLOAD_PAIR_COLLISION_MARGIN",
                                "0.0",
                            )
                            or 0.0
                        )
                        planned_payload_collision_pairs = (
                            _attached_payload_link_pair_collision_pairs(robot, planning_attached_obj)
                            if planned_payload_collision_validation_enabled
                            else []
                        )
                        planned_fk_admission_record["attached_payload_pair_collision_validation"] = {
                            "enabled": planned_payload_collision_validation_enabled,
                            "applied": False,
                            "margin": planned_payload_collision_margin,
                            "num_pairs": len(planned_payload_collision_pairs),
                        }
                        if not planned_fk_admission_applies:
                            if not planned_fk_admission_phase_in_range:
                                planned_fk_admission_record["reason"] = "phase_out_of_range"
                            elif phase_type != "coordinated":
                                planned_fk_admission_record["reason"] = "phase_type_not_coordinated"
                            elif len(planned_fk_admission_target_links) < 2:
                                planned_fk_admission_record["reason"] = "fewer_than_two_eef_targets"
                            elif not is_default_embodiment(emb_sel):
                                planned_fk_admission_record["reason"] = "not_default_embodiment"
                            elif len(success_idx) == 0:
                                planned_fk_admission_record["reason"] = "no_raw_success"
                        else:
                            accepted_success_idx = []
                            candidate_quality_enabled = bool(
                                int(os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_RANKING", "0") or 0)
                            )
                            candidate_quality_reject = bool(
                                int(os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_REJECT", "0") or 0)
                            )
                            candidate_quality_min_phase = int(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MIN_PHASE", "0") or 0
                            )
                            candidate_quality_max_phase = int(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MAX_PHASE", "999999") or 999999
                            )
                            candidate_quality_phase_in_range = (
                                candidate_quality_min_phase
                                <= int(env.execution_phase_ind)
                                <= candidate_quality_max_phase
                            )
                            candidate_quality_applies = (
                                candidate_quality_enabled
                                and candidate_quality_phase_in_range
                                and is_default_embodiment(emb_sel)
                            )
                            candidate_quality_max_base_path = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MAX_BASE_PATH_M", "0.8") or 0.8
                            )
                            candidate_quality_max_base_yaw_path = float(
                                os.environ.get(
                                    "MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MAX_BASE_YAW_PATH_RAD",
                                    "1.57079632679",
                                )
                                or 1.57079632679
                            )
                            candidate_quality_max_trunk_path = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MAX_TRUNK_PATH", "3.0") or 3.0
                            )
                            candidate_quality_max_arm_path = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_MAX_ARM_PATH", "5.0") or 5.0
                            )
                            candidate_quality_base_weight = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_BASE_WEIGHT", "4.0") or 4.0
                            )
                            candidate_quality_yaw_weight = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_YAW_WEIGHT", "1.0") or 1.0
                            )
                            candidate_quality_trunk_weight = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_TRUNK_WEIGHT", "1.0") or 1.0
                            )
                            candidate_quality_arm_weight = float(
                                os.environ.get("MOMAGEN_ARM_MP_CANDIDATE_QUALITY_ARM_WEIGHT", "1.0") or 1.0
                            )
                            downstream_staging_enabled = bool(
                                int(
                                    os.environ.get(
                                        "MOMAGEN_COORDINATED_MULTI_EE_DOWNSTREAM_CONTACT_STAGING_RANKING",
                                        "0",
                                    )
                                    or 0
                                )
                            )
                            downstream_staging_reject = bool(
                                int(
                                    os.environ.get(
                                        "MOMAGEN_COORDINATED_MULTI_EE_DOWNSTREAM_CONTACT_STAGING_REJECT",
                                        "0",
                                    )
                                    or 0
                                )
                            )
                            downstream_staging_weight = float(
                                os.environ.get(
                                    "MOMAGEN_COORDINATED_MULTI_EE_DOWNSTREAM_CONTACT_STAGING_WEIGHT",
                                    "10.0",
                                )
                                or 10.0
                            )
                            downstream_staging_max_dist = float(
                                os.environ.get(
                                    "MOMAGEN_COORDINATED_MULTI_EE_DOWNSTREAM_CONTACT_STAGING_MAX_DIST",
                                    "0.0",
                                )
                                or 0.0
                            )
                            downstream_staging_phase_in_range = (
                                int(os.environ.get("MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MIN_PHASE", "0") or 0)
                                <= int(env.execution_phase_ind)
                                <= int(
                                    os.environ.get(
                                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MAX_PHASE",
                                        "999999",
                                    )
                                    or 999999
                                )
                            )
                            downstream_staging_applies = bool(
                                downstream_staging_enabled
                                and downstream_staging_phase_in_range
                                and ref_obj is not None
                                and object_states.ToggledOn in getattr(ref_obj, "states", {})
                                and is_default_embodiment(emb_sel)
                            )
                            planned_fk_admission_record["candidate_quality_ranking"] = {
                                "enabled": candidate_quality_enabled,
                                "applied": bool(candidate_quality_applies),
                                "reject": candidate_quality_reject,
                                "phase_in_range": bool(candidate_quality_phase_in_range),
                                "min_phase": candidate_quality_min_phase,
                                "max_phase": candidate_quality_max_phase,
                                "thresholds": {
                                    "max_base_path_m": candidate_quality_max_base_path,
                                    "max_base_yaw_path_rad": candidate_quality_max_base_yaw_path,
                                    "max_trunk_path": candidate_quality_max_trunk_path,
                                    "max_arm_path": candidate_quality_max_arm_path,
                                },
                                "weights": {
                                    "base": candidate_quality_base_weight,
                                    "yaw": candidate_quality_yaw_weight,
                                    "trunk": candidate_quality_trunk_weight,
                                    "arm": candidate_quality_arm_weight,
                                },
                            }
                            planned_fk_admission_record["downstream_contact_staging_ranking"] = {
                                "enabled": downstream_staging_enabled,
                                "applied": bool(downstream_staging_applies),
                                "reject": downstream_staging_reject,
                                "phase_in_range": bool(downstream_staging_phase_in_range),
                                "weight": downstream_staging_weight,
                                "max_dist": downstream_staging_max_dist,
                            }

                            def _planned_fk_link_pose_local(robot_state, link_name):
                                link_key = str(link_name).split(":")[-1]
                                return robot_state.link_poses.get(link_key) or robot_state.link_poses.get(link_name)

                            def _planned_fk_quat_xyzw(planned_link_pose):
                                if planned_link_pose is None or not hasattr(planned_link_pose, "quaternion"):
                                    return None
                                quat_np = _debug_to_np(planned_link_pose.quaternion[-1])
                                if quat_np is None or len(quat_np) != 4:
                                    return None
                                # cuRobo FK reports wxyz; OmniGibson transform utils use xyzw.
                                return np.asarray([quat_np[1], quat_np[2], quat_np[3], quat_np[0]], dtype=float)

                            downstream_staging_context = None
                            if downstream_staging_applies:
                                try:
                                    toggle_state = ref_obj.states[object_states.ToggledOn]
                                    marker = getattr(toggle_state, "visual_marker", None)
                                    if marker is None:
                                        raise ValueError("missing_visual_marker")
                                    marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
                                    marker_pos_np = np.asarray(_debug_to_np(marker_pos_raw), dtype=float)
                                    marker_quat_np = np.asarray(_debug_to_np(marker_quat_raw), dtype=float)
                                    marker_rot_np = np.asarray(
                                        _debug_to_np(T.quat2mat(th.as_tensor(marker_quat_np, dtype=th.float32))),
                                        dtype=float,
                                    )
                                    marker_local_offset_raw = os.environ.get(
                                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MARKER_LOCAL_OFFSET",
                                        os.environ.get(
                                            "MOMAGEN_TOGGLE_MARKER_POST_MP_MARKER_LOCAL_OFFSET",
                                            os.environ.get(
                                                "MOMAGEN_TOGGLE_MARKER_TARGET_MARKER_LOCAL_OFFSET",
                                                "0.044,-0.035,0.013",
                                            ),
                                        ),
                                    )
                                    marker_local_offset_np = np.asarray(
                                        [
                                            float(value.strip())
                                            for value in marker_local_offset_raw.split(",")
                                            if value.strip()
                                        ],
                                        dtype=float,
                                    )
                                    if marker_local_offset_np.shape != (3,):
                                        raise ValueError(
                                            "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_MARKER_LOCAL_OFFSET "
                                            "must contain 3 comma-separated floats"
                                        )
                                    active_arms = [
                                        value.strip()
                                        for value in os.environ.get(
                                            "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_ACTIVE_ARMS",
                                            "left",
                                        ).split(",")
                                        if value.strip()
                                    ]
                                    hold_arms = [
                                        value.strip()
                                        for value in os.environ.get(
                                            "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_HOLD_ARMS",
                                            "right",
                                        ).split(",")
                                        if value.strip()
                                    ]
                                    active_arm = active_arms[0] if active_arms else "left"
                                    hold_arm = hold_arms[0] if hold_arms else None
                                    force_finger_link = os.environ.get(
                                        "MOMAGEN_TOGGLE_MARKER_CONTACT_PREALIGN_FORCE_FINGER_LINK",
                                        "",
                                    ).strip()
                                    active_finger_link = None
                                    for finger_link in getattr(robot, "finger_links", {}).get(active_arm, []):
                                        finger_name = getattr(finger_link, "name", str(finger_link))
                                        if not force_finger_link or force_finger_link == finger_name or force_finger_link in finger_name:
                                            active_finger_link = finger_link
                                            break
                                    if active_finger_link is None:
                                        raise ValueError("missing_active_finger_link")
                                    active_finger_body_name = getattr(
                                        active_finger_link,
                                        "body_name",
                                        str(getattr(active_finger_link, "name", active_finger_link)).split(":")[-1],
                                    )
                                    if not hold_arm or hold_arm not in robot.eef_link_names:
                                        raise ValueError("missing_hold_arm")
                                    hold_link_name = robot.eef_link_names[hold_arm]
                                    hold_pos, hold_quat = robot.get_eef_pose(hold_arm)
                                    hold_pos_np = np.asarray(_debug_to_np(hold_pos), dtype=float)
                                    hold_quat_np = np.asarray(_debug_to_np(hold_quat), dtype=float)
                                    hold_rot_np = np.asarray(
                                        _debug_to_np(T.quat2mat(th.as_tensor(hold_quat_np, dtype=th.float32))),
                                        dtype=float,
                                    )
                                    downstream_staging_context = {
                                        "active_arm": active_arm,
                                        "hold_arm": hold_arm,
                                        "hold_link_name": hold_link_name,
                                        "active_finger_body_name": active_finger_body_name,
                                        "marker_pos": marker_pos_np,
                                        "marker_quat": marker_quat_np,
                                        "marker_rot": marker_rot_np,
                                        "marker_local_offset": marker_local_offset_np,
                                        "marker_local_in_hold": hold_rot_np.T @ (marker_pos_np - hold_pos_np),
                                        "marker_rot_in_hold": hold_rot_np.T @ marker_rot_np,
                                    }
                                except Exception as exc:
                                    downstream_staging_context = None
                                    planned_fk_admission_record["downstream_contact_staging_ranking"][
                                        "error"
                                    ] = f"{type(exc).__name__}: {exc}"
                            try:
                                for raw_success_idx in success_idx.tolist():
                                    path = traj_paths[int(raw_success_idx)]
                                    robot_state = env.cmg.mg[emb_sel].kinematics.compute_kinematics(path)
                                    candidate_record = {
                                        "success_idx": int(raw_success_idx),
                                        "per_link": {},
                                        "failed_links": [],
                                        "passed": True,
                                    }
                                    for link_name in planned_fk_admission_target_links:
                                        target_pos_value = arm_mp_target_pos[link_name]
                                        if hasattr(target_pos_value, "ndim") and target_pos_value.ndim == 2:
                                            target_pos_value = target_pos_value[0]
                                        target_pos_np = _debug_to_np(target_pos_value)
                                        planned_link_pose = robot_state.link_poses.get(link_name)
                                        planned_pos_np = (
                                            None
                                            if planned_link_pose is None
                                            else _debug_to_np(planned_link_pose.position[-1])
                                        )
                                        link_record = {
                                            "target_pos": None if target_pos_np is None else target_pos_np.tolist(),
                                            "planned_pos": None if planned_pos_np is None else planned_pos_np.tolist(),
                                        }
                                        if (
                                            target_pos_np is None
                                            or planned_pos_np is None
                                            or not bool(np.isfinite(target_pos_np).all())
                                            or not bool(np.isfinite(planned_pos_np).all())
                                        ):
                                            link_record["error"] = "nonfinite_or_missing_position"
                                            candidate_record["failed_links"].append(link_name)
                                        else:
                                            pos_err = float(np.linalg.norm(planned_pos_np - target_pos_np))
                                            link_record["planned_to_target_pos_dist"] = pos_err
                                            if pos_err > planned_fk_admission_max_pos_err:
                                                candidate_record["failed_links"].append(link_name)
                                        candidate_record["per_link"][link_name] = link_record
                                    if planned_payload_collision_pairs:
                                        try:
                                            q_path = env.cmg.path_to_joint_trajectory(
                                                path, get_full_js=True, emb_sel=emb_sel
                                            )
                                            collision_result = env.cmg.check_link_pair_collisions(
                                                q_path,
                                                planned_payload_collision_pairs,
                                                initial_joint_pos=robot.get_joint_positions(),
                                                skip_obstacle_update=True,
                                                attached_obj=planning_attached_obj,
                                                attached_obj_scale=planning_attached_obj_scale,
                                                attached_obj_options=_attached_payload_options(planning_attached_obj),
                                                emb_sel=emb_sel,
                                                margin=planned_payload_collision_margin,
                                            )
                                            collision_np = _debug_to_np(collision_result.get("collision"))
                                            pair_records = []
                                            min_pair_dist = None
                                            for pair_record in collision_result.get("pairs", []):
                                                min_dist_np = _debug_to_np(pair_record.get("min_distance"))
                                                collision_pair_np = _debug_to_np(pair_record.get("collision"))
                                                if min_dist_np is not None:
                                                    finite_min_dist_np = np.asarray(min_dist_np)[
                                                        np.isfinite(np.asarray(min_dist_np))
                                                    ]
                                                    if finite_min_dist_np.size > 0:
                                                        pair_min = float(finite_min_dist_np.min())
                                                        min_pair_dist = (
                                                            pair_min
                                                            if min_pair_dist is None
                                                            else min(min_pair_dist, pair_min)
                                                        )
                                                pair_records.append(
                                                    {
                                                        "pair": pair_record.get("pair"),
                                                        "left_links": pair_record.get("left_links"),
                                                        "right_links": pair_record.get("right_links"),
                                                        "missing_links": pair_record.get("missing_links"),
                                                        "min_distance": None
                                                        if min_dist_np is None
                                                        else np.asarray(min_dist_np).tolist(),
                                                        "collision": None
                                                        if collision_pair_np is None
                                                        else np.asarray(collision_pair_np).astype(bool).tolist(),
                                                    }
                                                )
                                            payload_collision = bool(
                                                collision_np is not None
                                                and np.asarray(collision_np).astype(bool).any()
                                            )
                                            candidate_record["attached_payload_pair_collision"] = {
                                                "applied": True,
                                                "collision": payload_collision,
                                                "margin": planned_payload_collision_margin,
                                                "min_distance": min_pair_dist,
                                                "pairs": pair_records,
                                            }
                                            if payload_collision:
                                                candidate_record["failed_links"].append("attached_payload_pair_collision")
                                        except Exception as exc:
                                            candidate_record["attached_payload_pair_collision"] = {
                                                "applied": False,
                                                "error": f"{type(exc).__name__}: {exc}",
                                            }
                                            candidate_record["failed_links"].append("attached_payload_pair_collision_error")
                                    if candidate_quality_applies:
                                        try:
                                            q_path_for_quality = env.cmg.path_to_joint_trajectory(
                                                path, get_full_js=True, emb_sel=emb_sel
                                            )
                                            quality_by_group = _joint_path_quality_by_group(robot, q_path_for_quality)
                                            base_quality = quality_by_group.get("base", {})
                                            trunk_quality = quality_by_group.get("trunk", {})
                                            arm_left_quality = quality_by_group.get("arm_left", {})
                                            arm_right_quality = quality_by_group.get("arm_right", {})
                                            quality_failures = []
                                            base_path = float(base_quality.get("path_m", 0.0) or 0.0)
                                            base_yaw_path = float(base_quality.get("path_rad", 0.0) or 0.0)
                                            trunk_path = float(trunk_quality.get("path", 0.0) or 0.0)
                                            arm_left_path = float(arm_left_quality.get("path", 0.0) or 0.0)
                                            arm_right_path = float(arm_right_quality.get("path", 0.0) or 0.0)
                                            if base_path > candidate_quality_max_base_path:
                                                quality_failures.append("base_path")
                                            if base_yaw_path > candidate_quality_max_base_yaw_path:
                                                quality_failures.append("base_yaw_path")
                                            if trunk_path > candidate_quality_max_trunk_path:
                                                quality_failures.append("trunk_path")
                                            if max(arm_left_path, arm_right_path) > candidate_quality_max_arm_path:
                                                quality_failures.append("arm_path")
                                            quality_score = (
                                                candidate_quality_base_weight * base_path
                                                + candidate_quality_yaw_weight * base_yaw_path
                                                + candidate_quality_trunk_weight * trunk_path
                                                + candidate_quality_arm_weight * (arm_left_path + arm_right_path)
                                            )
                                            candidate_record["candidate_quality"] = {
                                                "applied": True,
                                                "score": float(quality_score),
                                                "failures": quality_failures,
                                                "by_group": quality_by_group,
                                            }
                                            if candidate_quality_reject and quality_failures:
                                                candidate_record["failed_links"].extend(
                                                    [f"candidate_quality_{name}" for name in quality_failures]
                                                )
                                        except Exception as exc:
                                            candidate_record["candidate_quality"] = {
                                                "applied": False,
                                                "error": f"{type(exc).__name__}: {exc}",
                                            }
                                            if candidate_quality_reject:
                                                candidate_record["failed_links"].append("candidate_quality_error")
                                    if downstream_staging_applies and downstream_staging_context is not None:
                                        try:
                                            hold_pose = _planned_fk_link_pose_local(
                                                robot_state,
                                                downstream_staging_context["hold_link_name"],
                                            )
                                            finger_pose = _planned_fk_link_pose_local(
                                                robot_state,
                                                downstream_staging_context["active_finger_body_name"],
                                            )
                                            hold_pos_np = None if hold_pose is None else _debug_to_np(hold_pose.position[-1])
                                            finger_pos_np = (
                                                None if finger_pose is None else _debug_to_np(finger_pose.position[-1])
                                            )
                                            hold_quat_np = _planned_fk_quat_xyzw(hold_pose)
                                            if (
                                                hold_pos_np is None
                                                or finger_pos_np is None
                                                or hold_quat_np is None
                                                or not bool(np.isfinite(hold_pos_np).all())
                                                or not bool(np.isfinite(finger_pos_np).all())
                                                or not bool(np.isfinite(hold_quat_np).all())
                                            ):
                                                raise ValueError("missing_or_nonfinite_planned_pose")
                                            hold_rot_np = np.asarray(
                                                _debug_to_np(T.quat2mat(th.as_tensor(hold_quat_np, dtype=th.float32))),
                                                dtype=float,
                                            )
                                            predicted_marker_pos = np.asarray(hold_pos_np, dtype=float) + hold_rot_np @ downstream_staging_context["marker_local_in_hold"]
                                            predicted_marker_rot = hold_rot_np @ downstream_staging_context["marker_rot_in_hold"]
                                            desired_finger_pos = (
                                                predicted_marker_pos
                                                + predicted_marker_rot @ downstream_staging_context["marker_local_offset"]
                                            )
                                            staging_dist = float(
                                                np.linalg.norm(np.asarray(finger_pos_np, dtype=float) - desired_finger_pos)
                                            )
                                            candidate_record["downstream_contact_staging"] = {
                                                "applied": True,
                                                "active_arm": downstream_staging_context["active_arm"],
                                                "hold_arm": downstream_staging_context["hold_arm"],
                                                "active_finger_body_name": downstream_staging_context[
                                                    "active_finger_body_name"
                                                ],
                                                "hold_link_name": downstream_staging_context["hold_link_name"],
                                                "planned_finger_pos": np.asarray(finger_pos_np, dtype=float).tolist(),
                                                "predicted_marker_pos": predicted_marker_pos.tolist(),
                                                "desired_finger_pos": desired_finger_pos.tolist(),
                                                "planned_finger_to_desired_dist": staging_dist,
                                                "score_bonus": float(downstream_staging_weight * staging_dist),
                                            }
                                            candidate_record.setdefault("candidate_quality", {}).setdefault(
                                                "score",
                                                0.0,
                                            )
                                            candidate_record["candidate_quality"][
                                                "score_before_downstream_contact_staging"
                                            ] = float(candidate_record["candidate_quality"]["score"])
                                            candidate_record["candidate_quality"]["score"] = float(
                                                candidate_record["candidate_quality"]["score"]
                                                + downstream_staging_weight * staging_dist
                                            )
                                            if (
                                                downstream_staging_reject
                                                and downstream_staging_max_dist > 0.0
                                                and staging_dist > downstream_staging_max_dist
                                            ):
                                                candidate_record["failed_links"].append(
                                                    "downstream_contact_staging_dist"
                                                )
                                        except Exception as exc:
                                            candidate_record["downstream_contact_staging"] = {
                                                "applied": False,
                                                "error": f"{type(exc).__name__}: {exc}",
                                            }
                                            if downstream_staging_reject:
                                                candidate_record["failed_links"].append(
                                                    "downstream_contact_staging_error"
                                                )
                                    candidate_record["passed"] = len(candidate_record["failed_links"]) == 0
                                    if candidate_record["passed"]:
                                        accepted_success_idx.append(int(raw_success_idx))
                                    else:
                                        planned_fk_failed_links_for_retry.extend(candidate_record["failed_links"])
                                    planned_fk_admission_record["candidates"].append(candidate_record)
                                if candidate_quality_applies and accepted_success_idx:
                                    accepted_candidate_records = [
                                        candidate
                                        for candidate in planned_fk_admission_record["candidates"]
                                        if int(candidate.get("success_idx", -1)) in set(accepted_success_idx)
                                    ]
                                    accepted_candidate_records.sort(
                                        key=lambda candidate: float(
                                            candidate.get("candidate_quality", {}).get("score", float("inf"))
                                        )
                                    )
                                    accepted_success_idx = [
                                        int(candidate["success_idx"]) for candidate in accepted_candidate_records
                                    ]
                                    planned_fk_admission_record["candidate_quality_ranking"][
                                        "sorted_accepted_success_idx"
                                    ] = accepted_success_idx
                                planned_fk_admission_record["attached_payload_pair_collision_validation"][
                                    "applied"
                                ] = bool(planned_payload_collision_pairs)
                                planned_fk_admission_record["accepted_success_idx"] = accepted_success_idx
                                success_idx = th.as_tensor(accepted_success_idx, dtype=th.long)
                            except Exception as exc:
                                planned_fk_admission_record["error"] = f"{type(exc).__name__}: {exc}"
                                planned_fk_admission_record["accepted_success_idx"] = []
                                planned_fk_admission_record["candidates"] = []
                                planned_fk_failed_links_for_retry = list(planned_fk_admission_target_links)
                                success_idx = th.as_tensor([], dtype=th.long)
                        phase_logs[env.execution_phase_ind].setdefault(
                            "coordinated_multi_ee_planned_fk_admission", []
                        ).append(planned_fk_admission_record)
                        print(
                            "[MOMAGEN_COORDINATED_MULTI_EE_PLANNED_FK_ADMISSION] "
                            + json.dumps(planned_fk_admission_record, default=str),
                            flush=True,
                        )

                    if len(success_idx) == 0:
                        print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                        if arm_mp_diag_enabled and not arm_mp_diag_done:
                            _run_arm_mp_failure_diagnostics(status_value, arm_mp_trial)
                            arm_mp_diag_done = True
                        switched_default_variant = False
                        for failed_link_name in planned_fk_failed_links_for_retry:
                            candidate_emb_sel = default_embodiment_variant(failed_link_name)
                            if (
                                candidate_emb_sel in getattr(env.cmg, "mg", {})
                                and candidate_emb_sel not in attempted_emb_sels
                            ):
                                retry_record = {
                                    "phase": int(env.execution_phase_ind),
                                    "trial": int(arm_mp_trial),
                                    "previous_emb_sel": str(emb_sel),
                                    "next_emb_sel": str(candidate_emb_sel),
                                    "failed_link": failed_link_name,
                                    "reason": "planned_fk_admission_failed",
                                }
                                phase_logs[env.execution_phase_ind].setdefault(
                                    "coordinated_multi_ee_default_variant_retry", []
                                ).append(retry_record)
                                print(
                                    "[MOMAGEN_COORDINATED_MULTI_EE_DEFAULT_VARIANT_RETRY] "
                                    + json.dumps(retry_record, default=str),
                                    flush=True,
                                )
                                emb_sel = candidate_emb_sel
                                attempted_emb_sels.add(str(emb_sel))
                                new_target_pos = copy.deepcopy(arm_mp_target_pos)
                                switched_default_variant = True
                                break
                        if switched_default_variant:
                            continue
                        if (
                            (not has_arm_no_torso_emb_sel)
                            and (not is_default_embodiment(emb_sel))
                            and (not default_emb_sel_fallback_used)
                            and (not disable_default_emb_sel_fallback)
                            and (CuRoboEmbodimentSelection.DEFAULT in getattr(env.cmg, "mg", {}))
                            and ("IK Fail" in status_value)
                        ):
                            print("Arm MP IK failed with ARM embodiment; retrying once with DEFAULT embodiment.")
                            emb_sel = CuRoboEmbodimentSelection.DEFAULT
                            default_emb_sel_fallback_used = True
                            new_target_pos = copy.deepcopy(arm_mp_target_pos)
                            continue
                        arm_mp_trial += 1
                        # modify target_pos a bit
                        for k in arm_mp_target_pos.keys():
                            new_target_pos[k] = arm_mp_target_pos[k] + th.rand(3) * 0.01 - 0.005
                        continue
                    else:
                        traj_path = traj_paths[success_idx[0]]
                        break

                print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                # ========================================================= End of Arm MP Planning ==========================================================

                # ========================================================== Arm MP Execution ==========================================================
                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)

                arm_mp_execution_start_time = time.time()

                # These lines are for debugging purposes.
                # successes, traj_paths = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=False, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)
                # full_result = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=True, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)

                # Convert planned joint trajectory to actions
                # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                arm_mp_debug = bool(int(os.environ.get("MOMAGEN_ARM_MP_DEBUG", "0") or 0))
                if arm_mp_debug:
                    q_delta = q_traj - q_traj[0:1]
                    print(
                        "[MOMAGEN_ARM_MP_DEBUG] "
                        f"emb_sel={emb_sel} raw_q_traj_shape={list(q_traj.shape)} "
                        f"finite={bool(th.isfinite(q_traj).all())} "
                        f"max_abs_delta={float(q_delta.abs().max())}",
                        flush=True,
                    )
                    for idx_name in ("base_idx", "trunk_control_idx", "arm_control_idx"):
                        if hasattr(robot, idx_name):
                            idx = getattr(robot, idx_name)
                            try:
                                idx_delta = q_delta[:, idx]
                                print(
                                    "[MOMAGEN_ARM_MP_DEBUG] "
                                    f"{idx_name}={idx} max_abs_delta={float(idx_delta.abs().max())}",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(f"[MOMAGEN_ARM_MP_DEBUG] failed_to_summarize_{idx_name}: {exc}", flush=True)
                # If we use curobo joint space planning instead of Cartesian space planning, we need to downsample the trajectory
                # q_traj = q_traj[::50]
                q_traj = _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01)
                max_q_traj_steps = int(os.environ.get("MOMAGEN_ARM_MP_MAX_Q_TRAJ_STEPS", "0") or 0)
                if max_q_traj_steps > 0 and q_traj.shape[0] > max_q_traj_steps:
                    keep_idx = th.linspace(0, q_traj.shape[0] - 1, max_q_traj_steps, device=q_traj.device).long()
                    q_traj = q_traj[keep_idx]
                    if arm_mp_debug:
                        print(
                            f"[MOMAGEN_ARM_MP_DEBUG] downsampled_interpolated_q_traj_shape={list(q_traj.shape)} "
                            f"max_steps={max_q_traj_steps}",
                            flush=True,
                        )
                elif arm_mp_debug:
                    print(f"[MOMAGEN_ARM_MP_DEBUG] interpolated_q_traj_shape={list(q_traj.shape)}", flush=True)
                q_traj = q_traj.cpu()
                cap_base_drift = bool(int(os.environ.get("MOMAGEN_ARM_MP_CAP_BASE_DRIFT", "0") or 0))
                cap_base_drift_record = None
                if cap_base_drift and is_default_embodiment(emb_sel) and hasattr(robot, "base_control_idx"):
                    cap_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_CAP_BASE_DRIFT_MIN_PHASE", "0") or 0)
                    cap_max_phase = int(
                        os.environ.get("MOMAGEN_ARM_MP_CAP_BASE_DRIFT_MAX_PHASE", "999999") or 999999
                    )
                    cap_phase_in_range = cap_min_phase <= int(env.execution_phase_ind) <= cap_max_phase
                    cap_max_xy = float(os.environ.get("MOMAGEN_ARM_MP_CAP_BASE_DRIFT_MAX_XY", "0.35") or 0.35)
                    cap_max_yaw = float(
                        os.environ.get("MOMAGEN_ARM_MP_CAP_BASE_DRIFT_MAX_YAW", "1.57079632679") or 1.57079632679
                    )
                    cap_base_drift_record = {
                        "enabled": True,
                        "phase": int(env.execution_phase_ind),
                        "phase_in_range": bool(cap_phase_in_range),
                        "emb_sel": str(emb_sel),
                        "max_xy": cap_max_xy,
                        "max_yaw": cap_max_yaw,
                    }
                    if cap_phase_in_range:
                        try:
                            base_control_idx = _safe_index_tensor(robot.base_control_idx)
                            base_start = robot.get_joint_positions().detach().cpu()[base_control_idx]
                            planned_base_before = q_traj[:, base_control_idx].detach().clone()
                            base_delta = planned_base_before - base_start.unsqueeze(0)
                            xy_norm = th.linalg.norm(base_delta[:, :2], dim=-1)
                            xy_scale = th.ones_like(xy_norm)
                            xy_mask = xy_norm > cap_max_xy
                            xy_scale[xy_mask] = cap_max_xy / xy_norm[xy_mask]
                            capped_base = planned_base_before.clone()
                            capped_base[:, :2] = base_start[:2].unsqueeze(0) + base_delta[:, :2] * xy_scale.unsqueeze(-1)
                            yaw_delta = th.as_tensor(
                                [float(wrap_angle(v)) for v in base_delta[:, 2]],
                                dtype=capped_base.dtype,
                                device=capped_base.device,
                            )
                            yaw_delta = th.clamp(yaw_delta, min=-cap_max_yaw, max=cap_max_yaw)
                            capped_base[:, 2] = base_start[2] + yaw_delta
                            q_traj[:, base_control_idx] = capped_base
                            before_final = planned_base_before[-1]
                            after_final = capped_base[-1]
                            cap_base_drift_record.update(
                                {
                                    "applied": True,
                                    "base_start": _debug_to_np(base_start).tolist(),
                                    "base_final_before": _debug_to_np(before_final).tolist(),
                                    "base_final_after": _debug_to_np(after_final).tolist(),
                                    "base_xy_drift_before": float(th.linalg.norm(before_final[:2] - base_start[:2])),
                                    "base_yaw_drift_before": float(abs(wrap_angle(before_final[2] - base_start[2]))),
                                    "base_xy_drift_after": float(th.linalg.norm(after_final[:2] - base_start[:2])),
                                    "base_yaw_drift_after": float(abs(wrap_angle(after_final[2] - base_start[2]))),
                                }
                            )
                        except Exception as exc:
                            cap_base_drift_record.update({"applied": False, "error": str(exc)})
                    else:
                        cap_base_drift_record.update({"applied": False, "reason": "phase_out_of_range"})
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_cap_base_drift", []).append(
                        cap_base_drift_record
                    )
                    print(
                        "[MOMAGEN_ARM_MP_CAP_BASE_DRIFT] "
                        + json.dumps(cap_base_drift_record, default=str),
                        flush=True,
                    )
                arm_mp_base_drift_record = None
                if hasattr(robot, "base_control_idx"):
                    try:
                        current_q_for_base_drift = robot.get_joint_positions().detach().cpu()
                        base_control_idx = _safe_index_tensor(robot.base_control_idx)
                        base_start = current_q_for_base_drift[base_control_idx]
                        base_final = q_traj[-1][base_control_idx]
                        base_xy_drift = float(th.linalg.norm(base_final[:2] - base_start[:2]))
                        base_yaw_drift = float(abs(wrap_angle(base_final[2] - base_start[2])))
                        arm_mp_base_drift_record = {
                            "phase": int(env.execution_phase_ind),
                            "emb_sel": str(emb_sel),
                            "base_start": _debug_to_np(base_start).tolist(),
                            "base_final": _debug_to_np(base_final).tolist(),
                            "base_xy_drift": base_xy_drift,
                            "base_yaw_drift": base_yaw_drift,
                            "target_keys": list(arm_mp_target_pos.keys()),
                        }
                        phase_logs[env.execution_phase_ind].setdefault("arm_mp_base_drift", []).append(
                            arm_mp_base_drift_record
                        )
                        print(
                            "[MOMAGEN_ARM_MP_BASE_DRIFT] "
                            + json.dumps(arm_mp_base_drift_record, default=str),
                            flush=True,
                        )
                    except Exception as exc:
                        arm_mp_base_drift_record = {
                            "phase": int(env.execution_phase_ind),
                            "emb_sel": str(emb_sel),
                            "error": str(exc),
                        }
                        phase_logs[env.execution_phase_ind].setdefault("arm_mp_base_drift", []).append(
                            arm_mp_base_drift_record
                        )
                        print(
                            "[MOMAGEN_ARM_MP_BASE_DRIFT] "
                            + json.dumps(arm_mp_base_drift_record, default=str),
                            flush=True,
                        )
                reject_base_drift = bool(int(os.environ.get("MOMAGEN_ARM_MP_REJECT_BASE_DRIFT", "0") or 0))
                reject_base_drift_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_REJECT_BASE_DRIFT_MIN_PHASE", "0") or 0)
                reject_base_drift_max_phase = int(
                    os.environ.get("MOMAGEN_ARM_MP_REJECT_BASE_DRIFT_MAX_PHASE", "999999") or 999999
                )
                reject_base_drift_phase_in_range = (
                    reject_base_drift_min_phase <= int(env.execution_phase_ind) <= reject_base_drift_max_phase
                )
                reject_base_drift_max_xy = float(os.environ.get("MOMAGEN_ARM_MP_REJECT_BASE_DRIFT_MAX_XY", "0.35") or 0.35)
                reject_base_drift_max_yaw = float(
                    os.environ.get("MOMAGEN_ARM_MP_REJECT_BASE_DRIFT_MAX_YAW", "1.57079632679") or 1.57079632679
                )
                if (
                    reject_base_drift
                    and reject_base_drift_phase_in_range
                    and arm_mp_base_drift_record is not None
                    and "error" not in arm_mp_base_drift_record
                    and (
                        arm_mp_base_drift_record["base_xy_drift"] > reject_base_drift_max_xy
                        or arm_mp_base_drift_record["base_yaw_drift"] > reject_base_drift_max_yaw
                    )
                ):
                    reject_record = dict(arm_mp_base_drift_record)
                    reject_record.update(
                        {
                            "enabled": True,
                            "reject": True,
                            "phase_in_range": True,
                            "min_phase": reject_base_drift_min_phase,
                            "max_phase": reject_base_drift_max_phase,
                            "max_xy": reject_base_drift_max_xy,
                            "max_yaw": reject_base_drift_max_yaw,
                        }
                    )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_base_drift_reject", []).append(
                        reject_record
                    )
                    print("[MOMAGEN_ARM_MP_BASE_DRIFT_REJECT] " + json.dumps(reject_record, default=str), flush=True)
                    env.err = "ArmMPBaseDriftRejected"
                    env.valid_env = False
                    env.execution_phase_ind += 1
                    return None
                if (
                    reject_base_drift
                    and not reject_base_drift_phase_in_range
                    and arm_mp_base_drift_record is not None
                    and "error" not in arm_mp_base_drift_record
                ):
                    reject_record = dict(arm_mp_base_drift_record)
                    reject_record.update(
                        {
                            "enabled": True,
                            "reject": False,
                            "phase_in_range": False,
                            "min_phase": reject_base_drift_min_phase,
                            "max_phase": reject_base_drift_max_phase,
                            "max_xy": reject_base_drift_max_xy,
                            "max_yaw": reject_base_drift_max_yaw,
                            "reason": "phase_out_of_range",
                        }
                    )
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_base_drift_reject", []).append(
                        reject_record
                    )
                    print("[MOMAGEN_ARM_MP_BASE_DRIFT_REJECT] " + json.dumps(reject_record, default=str), flush=True)
                execute_live_q_to_action = bool(
                    int(os.environ.get("MOMAGEN_ARM_MP_EXECUTE_LIVE_Q_TO_ACTION", "0") or 0)
                ) or bool(wholebody_arm_mp_enabled and is_default_embodiment(emb_sel))
                if execute_live_q_to_action and (len(left_mp_waypoints) == 0 or len(right_mp_waypoints) == 0):
                    # The live conversion path is intended for explicit whole-body manipulation planning where
                    # CuRobo controls all constrained EEFs and the holonomic base in one trajectory.  The legacy
                    # single-arm overlay path below precomputes replay IK actions for the inactive arm, so keep it
                    # on the old action-precompute path until that case is needed.
                    phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp_execution", []).append(
                        {
                            "live_q_to_action_requested": True,
                            "live_q_to_action_applied": False,
                            "reason": "single_arm_overlay_not_supported",
                            "left_mp_waypoints": int(len(left_mp_waypoints)),
                            "right_mp_waypoints": int(len(right_mp_waypoints)),
                        }
                    )
                    execute_live_q_to_action = False
                else:
                    phase_logs[env.execution_phase_ind].setdefault("wholebody_arm_mp_execution", []).append(
                        {
                            "live_q_to_action_requested": bool(
                                int(os.environ.get("MOMAGEN_ARM_MP_EXECUTE_LIVE_Q_TO_ACTION", "0") or 0)
                            ),
                            "live_q_to_action_applied": bool(execute_live_q_to_action),
                            "reason": None,
                            "emb_sel": str(emb_sel),
                        }
                    )
                arm_mp_tracking_diag_enabled = bool(int(os.environ.get("MOMAGEN_ARM_MP_TRACKING_DIAG", "0") or 0))
                arm_mp_tracking_diag_record = None
                if arm_mp_tracking_diag_enabled:
                    try:
                        def _pose7_to_dict(pose7):
                            pose_np = _debug_to_np(pose7)
                            if pose_np is None:
                                return None
                            return {
                                "pos": pose_np[:3].tolist(),
                                "quat": pose_np[3:7].tolist(),
                            }

                        def _target_pose_dict_by_link(target_pos_by_link, target_quat_by_link):
                            records = {}
                            for link_name, pos in target_pos_by_link.items():
                                quat = target_quat_by_link.get(link_name)
                                if quat is None:
                                    continue
                                pos_np = _debug_to_np(pos[0] if hasattr(pos, "ndim") and pos.ndim == 2 else pos)
                                quat_np = _debug_to_np(quat[0] if hasattr(quat, "ndim") and quat.ndim == 2 else quat)
                                records[link_name] = {
                                    "pos": None if pos_np is None else pos_np.tolist(),
                                    "quat": None if quat_np is None else quat_np.tolist(),
                                }
                            return records

                        def _actual_link_pose_dict_by_name(link_names):
                            records = {}
                            for link_name in link_names:
                                try:
                                    pos, quat = robot.links[link_name].get_position_orientation()
                                    records[link_name] = {
                                        "pos": _debug_to_np(pos).tolist(),
                                        "quat": _debug_to_np(quat).tolist(),
                                    }
                                except Exception as exc:
                                    records[link_name] = {"error": str(exc)}
                            return records

                        arm_mp_tracking_diag_record = {
                            "phase": int(env.execution_phase_ind),
                            "emb_sel": str(emb_sel),
                            "execute_live_q_to_action": bool(execute_live_q_to_action),
                            "q_traj_shape": list(q_traj.shape),
                            "target_keys": list(arm_mp_target_pos.keys()),
                            "primary_link_override": arm_mp_primary_link_override,
                        }
                        interesting_links = sorted(
                            set(arm_mp_target_pos.keys())
                            | set(robot.eef_link_names.values())
                            | set(getattr(env.cmg, "additional_links", {}).get(emb_sel, []))
                        )
                        arm_mp_tracking_diag_record["curobo_ee_link"] = str(
                            getattr(env.cmg, "ee_link", {}).get(emb_sel, None)
                        )
                        arm_mp_tracking_diag_record["curobo_additional_links"] = list(
                            getattr(env.cmg, "additional_links", {}).get(emb_sel, [])
                        )
                        try:
                            raw_robot_state = env.cmg.mg[emb_sel].kinematics.compute_kinematics(traj_path)
                            arm_mp_tracking_diag_record["curobo_raw_fk_final_link_pose"] = {
                                link_name: {
                                    "pos": _debug_to_np(poses.position[-1]).tolist(),
                                    "quat_wxyz": _debug_to_np(poses.quaternion[-1]).tolist(),
                                    "quat": _debug_to_np(poses.quaternion[-1][[1, 2, 3, 0]]).tolist(),
                                }
                                for link_name, poses in raw_robot_state.link_poses.items()
                                if link_name in interesting_links
                            }
                        except Exception as exc:
                            arm_mp_tracking_diag_record["curobo_raw_fk_error"] = str(exc)
                        try:
                            processed_joint_state = lazy.curobo.types.state.JointState(
                                position=env.cmg.tensor_args.to_device(q_traj),
                                joint_names=env.cmg.robot_joint_names,
                            ).get_ordered_joint_state(env.cmg.mg[emb_sel].kinematics.joint_names)
                            processed_robot_state = env.cmg.mg[emb_sel].kinematics.compute_kinematics(
                                processed_joint_state
                            )
                            arm_mp_tracking_diag_record["curobo_processed_fk_final_link_pose"] = {
                                link_name: {
                                    "pos": _debug_to_np(poses.position[-1]).tolist(),
                                    "quat_wxyz": _debug_to_np(poses.quaternion[-1]).tolist(),
                                    "quat": _debug_to_np(poses.quaternion[-1][[1, 2, 3, 0]]).tolist(),
                                }
                                for link_name, poses in processed_robot_state.link_poses.items()
                                if link_name in interesting_links
                            }
                        except Exception as exc:
                            arm_mp_tracking_diag_record["curobo_processed_fk_error"] = str(exc)
                        try:
                            planned_link_poses = env.cmg.path_to_eef_trajectory(
                                traj_path,
                                return_axisangle=False,
                                emb_sel=emb_sel,
                            )
                            arm_mp_tracking_diag_record["curobo_planned_final_link_pose"] = {
                                link_name: _pose7_to_dict(poses[-1])
                                for link_name, poses in planned_link_poses.items()
                                if link_name in interesting_links
                            }
                        except Exception as exc:
                            arm_mp_tracking_diag_record["curobo_planned_final_link_pose_error"] = str(exc)
                        arm_mp_tracking_diag_record["target_pose_by_link"] = _target_pose_dict_by_link(
                            arm_mp_target_pos,
                            arm_mp_target_quat,
                        )
                        arm_mp_tracking_diag_record["actual_link_pose_before_execution"] = _actual_link_pose_dict_by_name(
                            interesting_links
                        )
                        planned_final_q = q_traj[-1].detach().clone()
                        current_q = robot.get_joint_positions().detach().cpu()
                        arm_mp_tracking_diag_record["start_q_to_plan_final_max_abs"] = float(
                            (current_q - planned_final_q).abs().max()
                        )
                        arm_mp_tracking_diag_record["start_q_to_plan_final_by_group"] = _joint_error_summary_by_group(
                            robot, current_q, planned_final_q
                        )
                        for idx_name in ("base_idx", "trunk_control_idx", "arm_control_idx"):
                            if hasattr(robot, idx_name):
                                idx = _safe_index_tensor(getattr(robot, idx_name))
                                try:
                                    arm_mp_tracking_diag_record[f"{idx_name}_start_to_plan_final_max_abs"] = float(
                                        (current_q[idx] - planned_final_q[idx]).abs().max()
                                    )
                                except Exception as exc:
                                    arm_mp_tracking_diag_record[f"{idx_name}_start_to_plan_final_err"] = str(exc)
                    except Exception as exc:
                        arm_mp_tracking_diag_record = {"phase": int(env.execution_phase_ind), "error": str(exc)}

                arm_mp_explicit_target_links = set(arm_mp_target_pos.keys())
                left_has_explicit_arm_mp_target = robot.eef_link_names["left"] in arm_mp_explicit_target_links
                right_has_explicit_arm_mp_target = robot.eef_link_names["right"] in arm_mp_explicit_target_links
                arm_mp_inactive_arm_freeze_record = {
                    "phase": int(env.execution_phase_ind),
                    "target_keys": list(arm_mp_explicit_target_links),
                    "left_object_ref": object_ref.get("arm_left"),
                    "right_object_ref": object_ref.get("arm_right"),
                    "left_has_explicit_arm_mp_target": bool(left_has_explicit_arm_mp_target),
                    "right_has_explicit_arm_mp_target": bool(right_has_explicit_arm_mp_target),
                    "left_freeze_applied": bool(
                        object_ref["arm_left"] is None
                        and not _arm_has_active_payload("left")
                        and not left_has_explicit_arm_mp_target
                    ),
                    "right_freeze_applied": bool(
                        object_ref["arm_right"] is None
                        and not _arm_has_active_payload("right")
                        and not right_has_explicit_arm_mp_target
                    ),
                }
                phase_logs[env.execution_phase_ind].setdefault(
                    "arm_mp_inactive_arm_freeze", []
                ).append(arm_mp_inactive_arm_freeze_record)
                print(
                    "[MOMAGEN_ARM_MP_INACTIVE_ARM_FREEZE] "
                    + json.dumps(arm_mp_inactive_arm_freeze_record, default=str),
                    flush=True,
                )

                mp_actions = []
                for q_idx, j_pos in enumerate(q_traj):
                    if arm_mp_debug and q_idx % 250 == 0:
                        print(f"[MOMAGEN_ARM_MP_DEBUG] converting_q_idx={q_idx}/{len(q_traj)}", flush=True)

                    # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                    if (
                        object_ref["arm_left"] is None
                        and not _arm_has_active_payload("left")
                        and not left_has_explicit_arm_mp_target
                    ):
                        j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    elif (
                        object_ref["arm_right"] is None
                        and not _arm_has_active_payload("right")
                        and not right_has_explicit_arm_mp_target
                    ):
                        j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                    if execute_live_q_to_action:
                        # Defer q->action conversion until immediately before env.step. This is important when
                        # the planned trajectory includes holonomic base x/y/rz joints: the base controller action
                        # is expressed relative to the robot's current pose, which changes after every step.
                        action = j_pos.detach().clone()
                    else:
                        action = _joint_trajectory_point_to_action(robot, j_pos).cpu().numpy()

                    # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                    if (not execute_live_q_to_action) and left_gripper_action is not None:
                        action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                    if (not execute_live_q_to_action) and right_gripper_action is not None:
                        action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]

                    mp_actions.append(action)

                left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                if len(left_mp_waypoints) == 0:
                    # We need to pad the left hand waypoints to match the length of the MP trajectory
                    if len(left_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                            left_replay_waypoints.append(last_waypoint)

                    left_eef_poses = []
                    # We convert the target pose of the left hand to replay_action
                    # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                        left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_left"]
                        action[action_idx] = replay_action[action_idx]
                        action[env_interface.gripper_action_dim[0]] = left_replay_waypoints[i].gripper_action[0]

                    # We remove the waypoints that have been replayed for the left arm
                    left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                # Same logic as above but for the right hand
                elif len(right_mp_waypoints) == 0:
                    if len(right_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                            right_replay_waypoints.append(last_waypoint)
                    right_eef_poses = []
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                        right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_right"]
                        action[action_idx] = replay_action[action_idx]
                        action[env_interface.gripper_action_dim[1]] = right_replay_waypoints[i].gripper_action[1]

                    right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                init_global_env_step = env.global_env_step
                num_repeat = 1
                arm_mp_exec_debug_interval = int(
                    os.environ.get("MOMAGEN_ARM_MP_EXEC_DEBUG_INTERVAL", "250") or 250
                )
                for i, mp_action in enumerate(mp_actions):
                    if arm_mp_debug and arm_mp_exec_debug_interval > 0 and i % arm_mp_exec_debug_interval == 0:
                        print(f"[MOMAGEN_ARM_MP_DEBUG] executing_mp_action_idx={i}/{len(mp_actions)}", flush=True)
                    for _ in range(num_repeat):
                        if execute_live_q_to_action:
                            mp_action = _joint_trajectory_point_to_action(robot, mp_action).cpu().numpy()
                            if left_gripper_action is not None:
                                mp_action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                            if right_gripper_action is not None:
                                mp_action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        # TODO: Check if we can use primitive stack execute action here. This will allow for checking convergence errors etc.
                        if arm_mp_tracking_diag_record is not None and i in {0, len(mp_actions) - 1}:
                            arm_mp_tracking_diag_record.setdefault("action_limit_summary", {})[
                                f"pre_postprocess_step_{i}"
                            ] = _action_limit_summary_by_controller(robot, mp_action)
                        mp_action = _postprocess_action_compatible(env, mp_action)
                        if arm_mp_tracking_diag_record is not None and i in {0, len(mp_actions) - 1}:
                            arm_mp_tracking_diag_record.setdefault("action_limit_summary", {})[
                                f"post_postprocess_step_{i}"
                            ] = _action_limit_summary_by_controller(robot, mp_action)
                        if arm_mp_debug and i < int(os.environ.get("MOMAGEN_ARM_MP_DEBUG_ACTION_STEPS", "3") or 3):
                            action_np = np.asarray(mp_action)
                            print(
                                "[MOMAGEN_MP_ACTION_DEBUG] "
                                f"i={i} emb_sel={emb_sel} finite={bool(np.isfinite(action_np).all())} "
                                f"min={float(np.nanmin(action_np))} max={float(np.nanmax(action_np))} "
                                f"norm={float(np.linalg.norm(np.nan_to_num(action_np)))}",
                                flush=True,
                            )
                            for controller_name, action_idx in robot.controller_action_idx.items():
                                try:
                                    print(
                                        "[MOMAGEN_MP_ACTION_DEBUG_SLICE] "
                                        f"i={i} controller={controller_name} idx={action_idx} "
                                        f"value={action_np[action_idx].tolist()}",
                                        flush=True,
                                    )
                                except Exception as exc:
                                    print(
                                        "[MOMAGEN_MP_ACTION_DEBUG_SLICE] "
                                        f"i={i} controller={controller_name} failed={exc}",
                                        flush=True,
                                    )
                            if not np.isfinite(action_np).all():
                                raise RuntimeError("non-finite mp_action before env.step")
                        if bool(int(os.environ.get("MOMAGEN_ABORT_BEFORE_ARM_MP_EXECUTION", "0") or 0)):
                            raise RuntimeError("MOMAGEN_ABORT_BEFORE_ARM_MP_EXECUTION before env.step")
                        env.step(mp_action, video_writer)
                        if enable_marker_vis:
                            env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                            env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                            env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                            env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                final_settle_steps = int(os.environ.get("MOMAGEN_ARM_MP_FINAL_SETTLE_STEPS", "0") or 0)
                final_settle_min_phase = int(os.environ.get("MOMAGEN_ARM_MP_FINAL_SETTLE_MIN_PHASE", "0") or 0)
                final_settle_max_phase = int(
                    os.environ.get("MOMAGEN_ARM_MP_FINAL_SETTLE_MAX_PHASE", "999999") or 999999
                )
                final_settle_phase_in_range = (
                    final_settle_min_phase <= int(env.execution_phase_ind) <= final_settle_max_phase
                )
                if final_settle_steps > 0:
                    final_settle_record = {
                        "enabled": True,
                        "applied": bool(final_settle_phase_in_range),
                        "phase": int(env.execution_phase_ind),
                        "emb_sel": str(emb_sel),
                        "steps": int(final_settle_steps),
                        "min_phase": final_settle_min_phase,
                        "max_phase": final_settle_max_phase,
                    }
                    if final_settle_phase_in_range:
                        try:
                            planned_final_q_for_settle = q_traj[-1].detach().cpu()
                            before_settle_q = robot.get_joint_positions().detach().cpu()
                            final_settle_record["start_q_to_plan_final_max_abs"] = float(
                                (before_settle_q - planned_final_q_for_settle).abs().max()
                            )
                            final_settle_record["start_q_to_plan_final_by_group"] = _joint_error_summary_by_group(
                                robot, before_settle_q, planned_final_q_for_settle
                            )
                            for idx_name in ("base_idx", "trunk_control_idx", "arm_control_idx"):
                                if hasattr(robot, idx_name):
                                    idx = _safe_index_tensor(getattr(robot, idx_name))
                                    try:
                                        final_settle_record[f"{idx_name}_start_to_plan_final_max_abs"] = float(
                                            (before_settle_q[idx] - planned_final_q_for_settle[idx]).abs().max()
                                        )
                                    except Exception as exc:
                                        final_settle_record[f"{idx_name}_start_to_plan_final_err"] = str(exc)
                            for settle_step in range(final_settle_steps):
                                settle_action = _joint_trajectory_point_to_action(
                                    robot, planned_final_q_for_settle.detach().clone()
                                ).cpu().numpy()
                                if left_gripper_action is not None:
                                    settle_action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                                if right_gripper_action is not None:
                                    settle_action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                                state = env.get_state()["states"]
                                obs, obs_info = env.get_obs_IL()
                                datagen_info = env_interface.get_datagen_info(action=settle_action)
                                if settle_step in {0, final_settle_steps - 1}:
                                    final_settle_record.setdefault("action_limit_summary", {})[
                                        f"pre_postprocess_step_{settle_step}"
                                    ] = _action_limit_summary_by_controller(robot, settle_action)
                                settle_action = _postprocess_action_compatible(env, settle_action)
                                if settle_step in {0, final_settle_steps - 1}:
                                    final_settle_record.setdefault("action_limit_summary", {})[
                                        f"post_postprocess_step_{settle_step}"
                                    ] = _action_limit_summary_by_controller(robot, settle_action)
                                env.step(settle_action, video_writer)
                                if enable_marker_vis:
                                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                                    if left_eef_poses:
                                        env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[-1])
                                    if right_eef_poses:
                                        env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[-1])
                                local_env_step += 1
                                env.global_env_step += 1
                                states.append(state)
                                actions.append(settle_action)
                                observations.append(obs)
                                observations_info.append(json.dumps(obs_info))
                                datagen_infos.append(datagen_info)
                                cur_success_metrics = env.is_success()
                                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                                for k in success:
                                    success[k] = success[k] or cur_success_metrics[k]
                            after_settle_q = robot.get_joint_positions().detach().cpu()
                            final_settle_record["executed_q_to_plan_final_max_abs"] = float(
                                (after_settle_q - planned_final_q_for_settle).abs().max()
                            )
                            final_settle_record["executed_q_to_plan_final_by_group"] = _joint_error_summary_by_group(
                                robot, after_settle_q, planned_final_q_for_settle
                            )
                            for idx_name in ("base_idx", "trunk_control_idx", "arm_control_idx"):
                                if hasattr(robot, idx_name):
                                    idx = _safe_index_tensor(getattr(robot, idx_name))
                                    try:
                                        final_settle_record[f"{idx_name}_executed_to_plan_final_max_abs"] = float(
                                            (after_settle_q[idx] - planned_final_q_for_settle[idx]).abs().max()
                                        )
                                    except Exception as exc:
                                        final_settle_record[f"{idx_name}_executed_to_plan_final_err"] = str(exc)
                        except Exception as exc:
                            final_settle_record["applied"] = False
                            final_settle_record["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        final_settle_record["reason"] = "phase_out_of_range"
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_final_settle", []).append(
                        final_settle_record
                    )
                    print("[MOMAGEN_ARM_MP_FINAL_SETTLE] " + json.dumps(final_settle_record, default=str), flush=True)

                if arm_mp_tracking_diag_record is not None:
                    try:
                        final_q = robot.get_joint_positions().detach().cpu()
                        planned_final_q = q_traj[-1].detach().cpu()
                        arm_mp_tracking_diag_record["executed_q_to_plan_final_max_abs"] = float(
                            (final_q - planned_final_q).abs().max()
                        )
                        arm_mp_tracking_diag_record["executed_q_to_plan_final_by_group"] = _joint_error_summary_by_group(
                            robot, final_q, planned_final_q
                        )
                        for idx_name in ("base_idx", "trunk_control_idx", "arm_control_idx"):
                            if hasattr(robot, idx_name):
                                idx = _safe_index_tensor(getattr(robot, idx_name))
                                try:
                                    arm_mp_tracking_diag_record[f"{idx_name}_executed_to_plan_final_max_abs"] = float(
                                        (final_q[idx] - planned_final_q[idx]).abs().max()
                                    )
                                except Exception as exc:
                                    arm_mp_tracking_diag_record[f"{idx_name}_executed_to_plan_final_err"] = str(exc)
                        arm_mp_tracking_diag_record["robot_base_pose_after"] = {
                            "pos": _debug_to_np(robot.get_position_orientation()[0]).tolist(),
                            "quat": _debug_to_np(robot.get_position_orientation()[1]).tolist(),
                        }
                        arm_mp_tracking_diag_record["eef_pos_after"] = {
                            arm_name: _debug_to_np(robot.eef_links[arm_name].get_position_orientation()[0]).tolist()
                            for arm_name in ("left", "right")
                        }
                        planned_pose_key = (
                            "curobo_processed_fk_final_link_pose"
                            if "curobo_processed_fk_final_link_pose" in arm_mp_tracking_diag_record
                            else "curobo_planned_final_link_pose"
                            if "curobo_planned_final_link_pose" in arm_mp_tracking_diag_record
                            else "curobo_raw_fk_final_link_pose"
                            if "curobo_raw_fk_final_link_pose" in arm_mp_tracking_diag_record
                            else None
                        )
                        if planned_pose_key is not None:
                            target_pose_by_link = arm_mp_tracking_diag_record.get("target_pose_by_link", {})
                            actual_after = _actual_link_pose_dict_by_name(
                                arm_mp_tracking_diag_record[planned_pose_key].keys()
                            )
                            arm_mp_tracking_diag_record["actual_link_pose_after_execution"] = actual_after
                            link_errors = {}
                            for link_name, planned_pose in arm_mp_tracking_diag_record[planned_pose_key].items():
                                target_pose = target_pose_by_link.get(link_name)
                                actual_pose = actual_after.get(link_name)
                                link_errors[link_name] = {}
                                if planned_pose is not None and target_pose is not None:
                                    link_errors[link_name]["planned_to_target_pos_dist"] = float(
                                        np.linalg.norm(
                                            np.asarray(planned_pose["pos"], dtype=float)
                                            - np.asarray(target_pose["pos"], dtype=float)
                                        )
                                    )
                                if planned_pose is not None and actual_pose is not None and "pos" in actual_pose:
                                    link_errors[link_name]["actual_to_planned_pos_dist"] = float(
                                        np.linalg.norm(
                                            np.asarray(actual_pose["pos"], dtype=float)
                                            - np.asarray(planned_pose["pos"], dtype=float)
                                        )
                                    )
                                if target_pose is not None and actual_pose is not None and "pos" in actual_pose:
                                    link_errors[link_name]["actual_to_target_pos_dist"] = float(
                                        np.linalg.norm(
                                            np.asarray(actual_pose["pos"], dtype=float)
                                            - np.asarray(target_pose["pos"], dtype=float)
                                        )
                                    )
                            arm_mp_tracking_diag_record["link_position_errors"] = link_errors
                    except Exception as exc:
                        arm_mp_tracking_diag_record["post_execution_error"] = str(exc)
                    phase_logs[env.execution_phase_ind].setdefault("arm_mp_tracking_diag", []).append(
                        arm_mp_tracking_diag_record
                    )
                    print("[MOMAGEN_ARM_MP_TRACKING_DIAG] " + json.dumps(arm_mp_tracking_diag_record, default=str), flush=True)

                coordinated_multi_ee_validation_enabled = bool(
                    int(os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION", "1") or 1)
                )
                coordinated_multi_ee_validation_min_phase = int(
                    os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION_MIN_PHASE", "0") or 0
                )
                coordinated_multi_ee_validation_max_phase = int(
                    os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION_MAX_PHASE", "999999") or 999999
                )
                coordinated_multi_ee_validation_max_pos_err = float(
                    os.environ.get("MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION_MAX_POS_ERR", "0.05") or 0.05
                )
                coordinated_multi_ee_validation_phase_in_range = (
                    coordinated_multi_ee_validation_min_phase
                    <= int(env.execution_phase_ind)
                    <= coordinated_multi_ee_validation_max_phase
                )
                eef_target_link_names = [
                    link_name
                    for link_name in robot.eef_link_names.values()
                    if link_name in arm_mp_target_pos and link_name in arm_mp_target_quat
                ]
                coordinated_multi_ee_validation_applies = bool(
                    coordinated_multi_ee_validation_enabled
                    and coordinated_multi_ee_validation_phase_in_range
                    and phase_type == "coordinated"
                    and len(eef_target_link_names) >= 2
                )
                if coordinated_multi_ee_validation_enabled:
                    coordinated_multi_ee_validation_record = {
                        "enabled": True,
                        "applied": bool(coordinated_multi_ee_validation_applies),
                        "phase": int(env.execution_phase_ind),
                        "phase_type": phase_type,
                        "emb_sel": str(emb_sel),
                        "phase_in_range": bool(coordinated_multi_ee_validation_phase_in_range),
                        "min_phase": coordinated_multi_ee_validation_min_phase,
                        "max_phase": coordinated_multi_ee_validation_max_phase,
                        "max_pos_err": coordinated_multi_ee_validation_max_pos_err,
                        "explicit_target_links": list(arm_mp_target_pos.keys()),
                        "target_links": eef_target_link_names,
                        "per_link": {},
                        "failed_links": [],
                        "passed": True,
                    }
                    if not coordinated_multi_ee_validation_applies:
                        if not coordinated_multi_ee_validation_phase_in_range:
                            coordinated_multi_ee_validation_record["reason"] = "phase_out_of_range"
                        elif phase_type != "coordinated":
                            coordinated_multi_ee_validation_record["reason"] = "phase_type_not_coordinated"
                        elif len(eef_target_link_names) < 2:
                            coordinated_multi_ee_validation_record["reason"] = "fewer_than_two_eef_targets"
                    else:
                        try:
                            for link_name in eef_target_link_names:
                                target_pos = arm_mp_target_pos[link_name]
                                if hasattr(target_pos, "ndim") and target_pos.ndim == 2:
                                    target_pos = target_pos[0]
                                target_pos_np = _debug_to_np(target_pos)
                                actual_pos, actual_quat = robot.links[link_name].get_position_orientation()
                                actual_pos_np = _debug_to_np(actual_pos)
                                actual_quat_np = _debug_to_np(actual_quat)
                                link_record = {
                                    "target_pos": None if target_pos_np is None else target_pos_np.tolist(),
                                    "actual_pos": None if actual_pos_np is None else actual_pos_np.tolist(),
                                    "actual_quat": None if actual_quat_np is None else actual_quat_np.tolist(),
                                }
                                if (
                                    target_pos_np is None
                                    or actual_pos_np is None
                                    or not bool(np.isfinite(target_pos_np).all())
                                    or not bool(np.isfinite(actual_pos_np).all())
                                ):
                                    link_record["error"] = "nonfinite_or_missing_position"
                                    coordinated_multi_ee_validation_record["failed_links"].append(link_name)
                                else:
                                    pos_err = float(np.linalg.norm(actual_pos_np - target_pos_np))
                                    link_record["actual_to_target_pos_dist"] = pos_err
                                    if pos_err > coordinated_multi_ee_validation_max_pos_err:
                                        coordinated_multi_ee_validation_record["failed_links"].append(link_name)
                                coordinated_multi_ee_validation_record["per_link"][link_name] = link_record
                            coordinated_multi_ee_validation_record["passed"] = (
                                len(coordinated_multi_ee_validation_record["failed_links"]) == 0
                            )
                        except Exception as exc:
                            coordinated_multi_ee_validation_record["passed"] = False
                            coordinated_multi_ee_validation_record["error"] = f"{type(exc).__name__}: {exc}"
                            coordinated_multi_ee_validation_record["failed_links"] = list(eef_target_link_names)
                    phase_logs[env.execution_phase_ind].setdefault(
                        "coordinated_multi_ee_hard_validation", []
                    ).append(coordinated_multi_ee_validation_record)
                    print(
                        "[MOMAGEN_COORDINATED_MULTI_EE_HARD_VALIDATION] "
                        + json.dumps(coordinated_multi_ee_validation_record, default=str),
                        flush=True,
                    )
                    if (
                        coordinated_multi_ee_validation_record["applied"]
                        and not coordinated_multi_ee_validation_record["passed"]
                    ):
                        env.err = "CoordinatedMultiEEHardValidationFailed"
                        env.valid_env = False
                        env.execution_phase_ind += 1
                        return None

                local_env_step = _maybe_execute_toggle_marker_post_mp_press(
                    left_gripper_action=left_gripper_action,
                    right_gripper_action=right_gripper_action,
                    video_writer=video_writer,
                    states=states,
                    actions=actions,
                    observations=observations,
                    observations_info=observations_info,
                    datagen_infos=datagen_infos,
                    success=success,
                    local_env_step=local_env_step,
                    execute_live_q_to_action=execute_live_q_to_action,
                    wholebody_arm_mp_enabled=wholebody_arm_mp_enabled,
                    emb_sel=emb_sel,
                    press_timing_stage="post_mp",
                )

	                # # If using MP in default mode. Will remove this code later but keeping it for now for debugging purposes
	                # q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
	                # q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                # q_traj = q_traj.cpu()
                # left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(q_traj)
                # right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(q_traj)
                # num_repeat = 1
                # for i, j_pos in enumerate(q_traj):
                #     for _ in range(num_repeat):
                #         action = robot.q_to_action(j_pos).cpu().numpy()
                #         if left_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                #         if right_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                #         state = env.get_state()["states"]
                #         # obs, obs_info = env.get_obs_IL()
                #         datagen_info = env_interface.get_datagen_info(action=action)
                #         env.step(action)
                #         env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                #         env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                #         env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                #         env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                #         local_env_step += 1
                #         states.append(state)
                #         actions.append(action)
                #         observations.append(obs)
                #         datagen_infos.append(datagen_info)


            # Set the MP ranges to save to hdf5 file
            left_mp_ranges, right_mp_ranges = None, None
            if len(left_mp_waypoints) > 0:
                left_mp_ranges = [init_global_env_step, env.global_env_step]
            if len(right_mp_waypoints) > 0:
                right_mp_ranges = [init_global_env_step, env.global_env_step]


            MP_end_step_local = copy.deepcopy(local_env_step)
            # left MP points
            if len(left_mp_waypoints) == 0:
                left_MP_end_step_local = 0
            else:
                left_MP_end_step_local = MP_end_step_local
            if len(right_mp_waypoints) == 0:
                right_MP_end_step_local = 0
            else:
                right_MP_end_step_local = MP_end_step_local

            MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

            arm_mp_execution_finish_time = time.time()
            # Since there is only 1 trial for arm MP execution, we set the 0th index
            phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
            print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
            print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

            # ============================================== End of Arm MP Execution ==========================================================

            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)

            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()

            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos

            init_global_env_step = env.global_env_step
            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            for replay_step, (left_waypoint, right_waypoint) in enumerate(zip(left_replay_waypoints, right_replay_waypoints)):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None and not _arm_has_active_payload("right"):
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None and not _arm_has_active_payload("left"):
                    pose[:4, :] = current_left_ee_pose
                pose = _maybe_apply_toggle_marker_replay_correction(pose, replay_step=replay_step)
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                replay_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                replay_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3], dtype=th.float32)))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3], dtype=th.float32)))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            _log_manip_debug(
                stage="after_arm_replay",
                target_pos_by_link=target_pos if "target_pos" in locals() else None,
                attached_obj_by_link=attached_obj if isinstance(attached_obj, dict) else None,
            )

            contact_prealign_status = {}
            local_env_step = _maybe_execute_toggle_marker_contact_prealign(
                left_gripper_action=left_gripper_action,
                right_gripper_action=right_gripper_action,
                video_writer=video_writer,
                states=states,
                actions=actions,
                observations=observations,
                observations_info=observations_info,
                datagen_infos=datagen_infos,
                success=success,
                local_env_step=local_env_step,
                execute_live_q_to_action=execute_live_q_to_action,
                wholebody_arm_mp_enabled=wholebody_arm_mp_enabled,
                emb_sel=emb_sel,
                timing_stage="after_arm_replay",
                status_out=contact_prealign_status,
            )

            if contact_prealign_status.get("attempted") and contact_prealign_status.get("passed") is False:
                skip_record = {
                    "enabled": True,
                    "applied": False,
                    "phase": int(env.execution_phase_ind),
                    "timing_stage": "after_arm_replay",
                    "press_timing_stage": "after_arm_replay",
                    "reason": "contact_prealign_failed",
                    "contact_prealign_status": dict(contact_prealign_status),
                }
                phase_logs.setdefault(env.execution_phase_ind, {}).setdefault(
                    "toggle_marker_post_mp_press", []
                ).append(skip_record)
                print("[MOMAGEN_TOGGLE_MARKER_POST_MP_PRESS] " + json.dumps(skip_record, default=str), flush=True)
            else:
                local_env_step = _maybe_execute_toggle_marker_post_mp_press(
                    left_gripper_action=left_gripper_action,
                    right_gripper_action=right_gripper_action,
                    video_writer=video_writer,
                    states=states,
                    actions=actions,
                    observations=observations,
                    observations_info=observations_info,
                    datagen_infos=datagen_infos,
                    success=success,
                    local_env_step=local_env_step,
                    execute_live_q_to_action=execute_live_q_to_action,
                    wholebody_arm_mp_enabled=wholebody_arm_mp_enabled,
                    emb_sel=emb_sel,
                    press_timing_stage="after_arm_replay",
                )

            # =================================================== End of Arm Replay ==========================================================

            # =================================================== Arm/Torso Retract ==========================================================
            if retract_type != "no_retract":
                print("Starting Retract")

                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)

                retract_torso_only = False
                current_robot_base_pose_wrt_world = robot.get_position_orientation()
                # If we retract the left and right eef to the pose at the start of arm MP
                if retract_type == "retract_to_start_of_arm_mp":
                    if object_ref["arm_right"] is None:
                        arm_side = "left"
                        current_left_eef_pose = robot.get_eef_pose("left")
                        target_pos = {"left_eef_link": initial_left_eef_pose[0]}
                        # target_quat = {"left_eef_link": current_left_eef_pose[1]} # Retain current orientation
                        target_quat = {"left_eef_link": initial_left_eef_pose[1]} # Use initial orientation
                    elif object_ref["arm_left"] is None:
                        arm_side = "right"
                        current_right_eef_pose = robot.get_eef_pose("right")
                        target_pos = {"right_eef_link": initial_right_eef_pose[0]}
                        # target_quat = {"right_eef_link": current_right_eef_pose[1]} # Retain current orientation
                        target_quat = {"right_eef_link": initial_right_eef_pose[1]} # Retain initial orientation
                    # TODO: implement this. Not too important for now as this would never happen. In this case it's a bimanual coordinated and we don't need to retract
                    else:
                        pass

                # If we retract the left and right eef and eyes to a canonical pose
                elif retract_type == "retract_to_canonical_pose":
                    eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                    eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)

                    target_pos = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[0],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[0],
                        "eyes": eyes_reset_pose_wrt_world[0],
                    }
                    target_quat = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[1],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[1],
                        "eyes": eyes_reset_pose_wrt_world[1],
                    }

                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                    eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)
                    current_left_eef_pose = robot.get_eef_pose("left")

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)
                    current_right_eef_pose = robot.get_eef_pose("right")

                    target_pos = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[0],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[0],
                        "eyes": eyes_reset_pose_wrt_world[0],
                    }
                    target_quat = {
                        "left_eef_link": current_left_eef_pose[1],
                        "right_eef_link": current_right_eef_pose[1],
                        "eyes": eyes_reset_pose_wrt_world[1],
                    }

                else:
                    raise ValueError(f"Invalid retract type: {retract_type}")

                # Current R1 / R1Pro CuRobo configs do not expose an "eyes" link target.
                # MoMaGen's original retract path included it for head tracking, but head
                # tracking is Tiago-only in this OmniGibson version and causes CuRobo to
                # fail with KeyError('eyes').
                if not isinstance(robot, Tiago):
                    target_pos.pop("eyes", None)
                    target_quat.pop("eyes", None)

                # Aggregate target_pos and target_quat to match batch_size
                new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                new_target_quat = {
                    k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                }

                retval = self.obtain_attached_object(env, robot)
                grasp_action = retval["grasp_action"]
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]

                # if enable_marker_vis:
                #     if arm_side == "left":
                #         env.eef_goal_marker_left.set_position_orientation(target_pos["left_eef_link"], target_quat["left_eef_link"])
                #     elif arm_side == "right":
                #         env.eef_goal_marker_right.set_position_orientation(target_pos["right_eef_link"], target_quat["right_eef_link"])

                if retract_type == "retract_to_start_of_arm_mp":
                    emb_sel = getattr(
                        CuRoboEmbodimentSelection,
                        "ARM_NO_TORSO",
                        CuRoboEmbodimentSelection.ARM,
                    )
                elif retract_type == "retract_to_canonical_pose":
                    emb_sel = CuRoboEmbodimentSelection.ARM
                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    emb_sel = CuRoboEmbodimentSelection.ARM

                full_retract_mp_planning_start_time = time.time()
                mp_results, traj_paths = _compute_trajectories_with_paths(env.cmg,
                    target_pos=new_target_pos,
                    target_quat=new_target_quat,
                    is_local=False,
                    max_attempts=50,
                    timeout=20.0,
                    ik_fail_return=50,
                    enable_finetune_trajopt=True,
                    finetune_attempts=1,
                    return_full_result=True,
                    success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                    attached_obj=attached_obj,
                    attached_obj_scale=attached_obj_scale,
                    attached_obj_options=_attached_payload_options(attached_obj),
                    emb_sel=emb_sel,
                )
                full_retract_mp_planning_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0] = round(full_retract_mp_planning_finish_time - full_retract_mp_planning_start_time, 2)
                print("Time taken for full retract MP planning: ", phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0])

                successes = mp_results[0].success
                print("Retract Arm MP successes: ", successes)
                success_idx = th.where(successes)[0].cpu()

                if len(success_idx) == 0:
                    print(f"Arm retract failed with status {mp_results[0].status}.")
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = mp_results[0].status.value
                    retract_torso_only = True
                else:
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = "None"
                    full_retract_mp_execution_start_time = time.time()
                    traj_path = traj_paths[success_idx[0]]

                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01)
                    q_traj = q_traj.cpu()

                    num_repeat = 1
                    init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                    init_global_env_step = env.global_env_step
                    for j_pos in q_traj:
                        if retract_type == "retract_to_start_of_arm_mp":
                            if arm_side == "left":
                                j_pos[robot.arm_control_idx["right"]] = init_right_arm_pos
                            elif arm_side == "right":
                                j_pos[robot.arm_control_idx["left"]] = init_left_arm_pos

                        mp_action = _joint_trajectory_point_to_action(robot, j_pos).cpu().numpy()
                        mp_action[robot.gripper_action_idx["left"]] = grasp_action["left"]
                        mp_action[robot.gripper_action_idx["right"]] = grasp_action["right"]

                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        env.step(mp_action, video_writer)
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                    full_retract_mp_execution_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["full_retract_mp_execution_time"][0] = round(full_retract_mp_execution_finish_time - full_retract_mp_execution_start_time, 2)

                    num_phase_steps = env.global_env_step - init_global_env_step
                    for sensor_name, sensor in env.robot.sensors.items():
                        if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                            shortened_sensor_name = sensor_name.split(":")[1]
                            if num_phase_steps > 0:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                            else:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"]= 0
                            print(f"Visibility stats for full_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"])
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"]= 0
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_steps"] = num_phase_steps
                    print(f"Visibility stats for full_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"])

                # If full retract failed, try retracting only the torso
                if retract_torso_only and retract_type != "retract_to_start_of_arm_mp" and isinstance(robot, Tiago):
                    print("Retracting torso only")

                    # reset the visibility counter for each sensor
                    self.reset_visibility_counter(env)

                    target_pos = {"eyes": eyes_reset_pose_wrt_world[0]}
                    target_quat = {"eyes": eyes_reset_pose_wrt_world[1]}

                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                    new_target_quat = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()}

                    torso_retract_mp_planning_start_time = time.time()
                    mp_results, traj_paths = _compute_trajectories_with_paths(env.cmg,
                        target_pos=new_target_pos,
                        target_quat=new_target_quat,
                        is_local=False,
                        max_attempts=50,
                        timeout=20.0,
                        ik_fail_return=50,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1,
                        return_full_result=True,
                        success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                        attached_obj=attached_obj,
                        attached_obj_scale=attached_obj_scale,
                        attached_obj_options=_attached_payload_options(attached_obj),
                        emb_sel=emb_sel,
                    )
                    torso_retract_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["torso_retract_mp_planning_time"][0] = round(torso_retract_mp_planning_finish_time - torso_retract_mp_planning_start_time, 2)

                    successes = mp_results[0].success
                    print("Torso-only retract: Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()

                    if len(success_idx) == 0:
                        print(f"Torso retract failed with status {mp_results[0].status}.")
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = mp_results[0].status.value
                    else:
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = "None"
                        torso_retract_mp_execution_start_time = time.time()
                        traj_path = traj_paths[success_idx[0]]

                        q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                        q_traj = _add_linearly_interpolated_waypoints(env, q_traj, max_inter_dist=0.01)
                        q_traj = q_traj.cpu()

                        num_repeat = 1
                        init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                        init_global_env_step = env.global_env_step
                        for j_pos in q_traj:
                            mp_action = _joint_trajectory_point_to_action(robot, j_pos).cpu().numpy()
                            mp_action[robot.gripper_action_idx["left"]] = grasp_action["left"]
                            mp_action[robot.gripper_action_idx["right"]] = grasp_action["right"]
                            # Don't want to move the arm relative to the torso
                            mp_action[robot.arm_action_idx["right"]] = init_right_arm_pos
                            mp_action[robot.arm_action_idx["left"]] = init_left_arm_pos

                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            env.step(mp_action, video_writer)
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]

                        torso_retract_mp_execution_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_execution_time"][0] = round(torso_retract_mp_execution_finish_time - torso_retract_mp_execution_start_time, 2)

                        num_phase_steps = env.global_env_step - init_global_env_step
                        for sensor_name, sensor in env.robot.sensors.items():
                            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                                shortened_sensor_name = sensor_name.split(":")[1]
                                if num_phase_steps > 0:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                                else:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"]= 0
                                print(f"Visibility stats for torso_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"])
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"]= 0
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_steps"] = num_phase_steps
                        print(f"Visibility stats for torso_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"])

            # ================================================== End of Arm/Torso Retract ==========================================================

            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results
