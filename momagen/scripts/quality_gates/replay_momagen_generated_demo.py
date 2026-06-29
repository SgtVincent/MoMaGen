#!/usr/bin/env python3
"""Replay a MoMaGen generated HDF5 by restoring states[0] and stepping actions.

This is intentionally separate from BEHAVIOR DataPlaybackWrapper because MoMaGen
outputs `actions` / `states` / `env_args`, not DataCollectionWrapper's
`action` / `state` / `state_size` / `scene_file` schema.
"""
import argparse, json, os, sys, tempfile, traceback
from pathlib import Path
from copy import deepcopy

import h5py
import cv2
import imageio.v2 as imageio
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import ToggledOn
import omnigibson.utils.transform_utils as T

import momagen.utils.robomimic_utils as RobomimicUtils

R1PRO_OBS_CAMERA_KEYS = {
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
}

R1PRO_OBS_CAMERA_SENSOR_NAMES = {
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1:zed_link:Camera:0",
}


def _to_list(x):
    if x is None: return None
    if hasattr(x, 'detach'): x=x.detach().cpu().numpy()
    return np.asarray(x).tolist()


def _dist(a,b):
    return float(np.linalg.norm(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)))


def _task_success(env):
    try:
        s=env.is_success()
        return {k: bool(v) for k,v in s.items()}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def _state_error(ref_state):
    try:
        cur=_to_list(og.sim.dump_state(serialized=True))
        ref=np.asarray(ref_state, dtype=float)
        cur=np.asarray(cur, dtype=float)
        if cur.shape != ref.shape:
            return {'shape_match': False, 'current_shape': list(cur.shape), 'ref_shape': list(ref.shape)}
        delta=cur-ref
        return {
            'shape_match': True,
            'l2': float(np.linalg.norm(delta)),
            'mean_abs': float(np.mean(np.abs(delta))),
            'max_abs': float(np.max(np.abs(delta))),
        }
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def _snapshot(env, step, ref_state=None, target_objects=None):
    out={'step': int(step), 'success': _task_success(env), 'objects': {}}
    if ref_state is not None:
        out['state_error_vs_hdf5']=_state_error(ref_state)
    og_env=getattr(env,'env',env)
    robot=og_env.robots[0] if getattr(og_env,'robots',None) else None
    target_objects=set(target_objects or [])
    for obj in getattr(og_env.scene,'objects',[]):
        if target_objects and obj.name not in target_objects:
            continue
        if ToggledOn not in getattr(obj,'states',{}):
            continue
        state=obj.states[ToggledOn]
        marker=getattr(state,'visual_marker',None)
        marker_pos=None; marker_quat=None; marker_radius=None; overlap_pos=None; overlap_local_offset=None
        if marker is not None:
            marker_pose = marker.get_position_orientation()
            marker_pos=np.asarray(_to_list(marker_pose[0]), dtype=float)
            marker_quat=np.asarray(_to_list(marker_pose[1]), dtype=float)
            overlap_pos = marker_pos
            try:
                marker_radius=float(np.min(np.asarray(_to_list(marker.extent),dtype=float)*np.asarray(_to_list(marker.scale),dtype=float)))
            except Exception as e:
                marker_radius=f'ERR:{type(e).__name__}: {e}'
        finger_contact_objs=getattr(ToggledOn,'_finger_contact_objs',None)
        try:
            obj_in_contact=None if finger_contact_objs is None else obj in finger_contact_objs
        except Exception as e:
            obj_in_contact=f'ERR:{type(e).__name__}: {e}'
        row={
            'value': bool(state.get_value()),
            'robot_can_toggle_steps': int(getattr(state,'robot_can_toggle_steps',-1)),
            'obj_in_finger_contact_objs': obj_in_contact,
            'marker_pos': _to_list(marker_pos),
            'marker_quat': _to_list(marker_quat),
            'overlap_pos': _to_list(overlap_pos),
            'overlap_local_offset': overlap_local_offset,
            'marker_radius': marker_radius,
            'finger_min_dist_to_marker': {},
            'finger_min_dist_to_overlap': {},
            'overlap_primary_hit': None,
            'overlap_first_hit_radius': None,
            'overlap_probes': [],
        }
        if robot is not None and overlap_pos is not None:
            finger_paths={getattr(link,'prim_path',None) for links in getattr(robot,'finger_links',{}).values() for link in links}
            finger_paths.discard(None)
            row['finger_paths'] = sorted(finger_paths)
            marker_rot = None
            if marker_quat is not None:
                marker_rot = np.asarray(_to_list(T.quat2mat(th.as_tensor(marker_quat, dtype=th.float32))), dtype=float)
            for arm,links in getattr(robot,'finger_links',{}).items():
                vals=[]; overlap_vals=[]
                for link in links:
                    try:
                        link_pos = np.asarray(_to_list(link.get_position_orientation()[0]), dtype=float)
                        marker_delta = link_pos - marker_pos
                        overlap_delta = link_pos - overlap_pos
                        base = {
                            'link': getattr(link,'name',str(link)),
                            'prim_path': getattr(link,'prim_path',None),
                            'pos': _to_list(link_pos),
                        }
                        marker_row = {
                            **base,
                            'dist': float(np.linalg.norm(marker_delta)),
                            'delta_world': _to_list(marker_delta),
                        }
                        overlap_row = {
                            **base,
                            'dist': float(np.linalg.norm(overlap_delta)),
                            'delta_world': _to_list(overlap_delta),
                        }
                        if marker_rot is not None:
                            marker_row['delta_marker_local'] = _to_list(marker_rot.T @ marker_delta)
                            overlap_row['delta_marker_local'] = _to_list(marker_rot.T @ overlap_delta)
                        vals.append(marker_row)
                        overlap_vals.append(overlap_row)
                    except Exception as e:
                        vals.append({'error': f'{type(e).__name__}: {e}'})
                        overlap_vals.append({'error': f'{type(e).__name__}: {e}'})
                vals=[v for v in vals if 'dist' in v]
                overlap_vals=[v for v in overlap_vals if 'dist' in v]
                row['finger_min_dist_to_marker'][arm]=min(vals,key=lambda x:x['dist']) if vals else None
                row['finger_min_dist_to_overlap'][arm]=min(overlap_vals,key=lambda x:x['dist']) if overlap_vals else None
            radii=[marker_radius,0.03,0.05,0.075,0.10,0.125,0.15] if isinstance(marker_radius,float) else [0.03,0.05,0.075,0.10,0.125,0.15]
            seen=[]
            for radius in radii:
                if radius in seen: continue
                seen.append(radius)
                valid=False; hits=[]
                num_hits=0
                def report(hit):
                    nonlocal valid, num_hits
                    num_hits += 1
                    rb=str(getattr(hit,'rigid_body',''))
                    is_finger=rb in finger_paths
                    valid = valid or is_finger
                    if len(hits)<64: hits.append({'rigid_body':rb,'is_robot_finger':is_finger})
                    return True
                try:
                    og.sim.psqi.overlap_sphere(radius=float(radius), pos=overlap_pos.tolist(), reportFn=report)
                    row['overlap_probes'].append({
                        'radius': float(radius),
                        'valid_robot_finger_hit': bool(valid),
                        'num_hits': int(num_hits),
                        'hits': hits,
                    })
                    if radius == marker_radius:
                        row['overlap_primary_hit']=bool(valid)
                    if valid and row['overlap_first_hit_radius'] is None:
                        row['overlap_first_hit_radius']=float(radius)
                except Exception as e:
                    row.setdefault('overlap_errors',[]).append(f'{radius}:{type(e).__name__}: {e}')
        out['objects'][obj.name]=row
    return out


def summarize(records):
    summary={
        'num_records': len(records),
        'first_task_success_step': None,
        'first_toggle_value_step': None,
        'first_can_toggle_step': None,
        'first_primary_overlap_step': None,
        'max_robot_can_toggle_steps': 0,
        'best_left_finger_dist': None,
        'state_error': {'max_abs_max': None, 'mean_abs_max': None, 'l2_max': None, 'shape_mismatch_steps': []},
        'observation_visibility': {},
    }
    best=float('inf')
    for r in records:
        step=r['step']
        if r.get('success',{}).get('task') is True and summary['first_task_success_step'] is None:
            summary['first_task_success_step']=step
        err=r.get('state_error_vs_hdf5')
        if isinstance(err,dict):
            if err.get('shape_match') is False:
                summary['state_error']['shape_mismatch_steps'].append(step)
            for key in ('max_abs','mean_abs','l2'):
                if isinstance(err.get(key),(int,float)):
                    out_key=f'{key}_max'
                    prev=summary['state_error'].get(out_key)
                    summary['state_error'][out_key]=float(err[key]) if prev is None else max(float(prev), float(err[key]))
        for obj,row in r.get('objects',{}).items():
            can=row.get('robot_can_toggle_steps') or 0
            summary['max_robot_can_toggle_steps']=max(summary['max_robot_can_toggle_steps'], can)
            if row.get('value') and summary['first_toggle_value_step'] is None:
                summary['first_toggle_value_step']=step
            if can>0 and summary['first_can_toggle_step'] is None:
                summary['first_can_toggle_step']=step
            if row.get('overlap_primary_hit') and summary['first_primary_overlap_step'] is None:
                summary['first_primary_overlap_step']=step
            left=(row.get('finger_min_dist_to_overlap') or row.get('finger_min_dist_to_marker') or {}).get('left')
            if isinstance(left,dict) and isinstance(left.get('dist'),(int,float)) and left['dist']<best:
                best=left['dist']; summary['best_left_finger_dist']={'step':step,'object':obj,'dist':left['dist'],'link':left.get('link')}
            for cam_name, vis in (row.get('observation_visibility') or {}).items():
                cam_summary = summary['observation_visibility'].setdefault(
                    cam_name,
                    {
                        'frames': 0,
                        'object_visible_frames': 0,
                        'marker_in_frame_frames': 0,
                        'marker_with_object_visible_frames': 0,
                        'max_object_pixel_fraction': 0.0,
                        'max_object_bbox_area_fraction': 0.0,
                        'object_pixel_fraction_sum': 0.0,
                    },
                )
                cam_summary['frames'] += 1
                if vis.get('object_visible'):
                    cam_summary['object_visible_frames'] += 1
                if vis.get('marker_projection', {}).get('in_frame'):
                    cam_summary['marker_in_frame_frames'] += 1
                if vis.get('object_visible') and vis.get('marker_projection', {}).get('in_frame'):
                    cam_summary['marker_with_object_visible_frames'] += 1
                pix_frac = vis.get('object_pixel_fraction')
                if isinstance(pix_frac, (int, float)):
                    cam_summary['object_pixel_fraction_sum'] += float(pix_frac)
                    cam_summary['max_object_pixel_fraction'] = max(cam_summary['max_object_pixel_fraction'], float(pix_frac))
                bbox_frac = vis.get('object_bbox_area_fraction')
                if isinstance(bbox_frac, (int, float)):
                    cam_summary['max_object_bbox_area_fraction'] = max(cam_summary['max_object_bbox_area_fraction'], float(bbox_frac))
    for cam_summary in summary['observation_visibility'].values():
        frames = max(1, cam_summary['frames'])
        cam_summary['object_visible_rate'] = cam_summary['object_visible_frames'] / frames
        cam_summary['marker_in_frame_rate'] = cam_summary['marker_in_frame_frames'] / frames
        cam_summary['marker_with_object_visible_rate'] = cam_summary['marker_with_object_visible_frames'] / frames
        cam_summary['mean_object_pixel_fraction'] = cam_summary['object_pixel_fraction_sum'] / frames
        del cam_summary['object_pixel_fraction_sum']
    return summary


def write_payload(path, payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp_path=f"{path}.tmp"
    Path(tmp_path).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    os.replace(tmp_path,path)


def _camera_quat_looking_at(cam_pos, target):
    forward = np.asarray(target, dtype=np.float32) - np.asarray(cam_pos, dtype=np.float32)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    corrected_up = np.cross(right, forward)
    corrected_up = corrected_up / max(float(np.linalg.norm(corrected_up)), 1e-6)
    # USD camera looks along local -Z; columns are local axes in world frame.
    rot = th.tensor(np.stack([right, corrected_up, -forward], axis=1), dtype=th.float32)
    return T.mat2quat(rot)


def _set_review_camera(env, target_objects=None):
    if not target_objects:
        return
    og_env = getattr(env, 'env', env)
    target_obj = None
    target_names = set(target_objects)
    for obj in getattr(og_env.scene, 'objects', []):
        if obj.name in target_names:
            target_obj = obj
            break
    if target_obj is None:
        return
    target_pos = _to_list(target_obj.get_position_orientation()[0])
    target = np.asarray(target_pos, dtype=np.float32) + np.asarray([0.0, 0.0, 0.25], dtype=np.float32)
    cam_pos = target + np.asarray([-1.15, -1.25, 0.85], dtype=np.float32)
    quat = _camera_quat_looking_at(cam_pos, target)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(cam_pos, dtype=th.float32),
        orientation=quat,
    )


def _capture_frame(env, target_objects=None):
    _set_review_camera(env, target_objects=target_objects)
    for _ in range(3):
        og.sim.render()
    frame = og.sim.viewer_camera.get_obs()[0]['rgb']
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if frame.max(initial=0) <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[..., :3]


def _to_numpy_rgb(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    frame = np.asarray(x)
    if frame.ndim < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if float(frame.max(initial=0.0)) <= 1.5 else 1.0
            frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _find_obs_key(obs, key):
    if key in obs:
        return key
    suffix = f"{key.split('::')[0]}::rgb"
    for candidate in obs:
        if candidate.endswith(suffix):
            return candidate
    return None


def _ensure_camera_modalities(env, include_visibility=False):
    modalities = ["rgb"]
    if include_visibility:
        modalities.extend(["seg_instance", "seg_semantic"])
    og_env = getattr(env, "env", env)
    robot = og_env.robots[0] if getattr(og_env, "robots", None) else None
    if robot is None:
        return
    for sensor_name, sensor in getattr(robot, "sensors", {}).items():
        if "Camera" not in sensor_name:
            continue
        for modality in modalities:
            if hasattr(sensor, "add_modality") and modality not in getattr(sensor, "modalities", set()):
                sensor.add_modality(modality)


def _get_obs(env):
    for _ in range(3):
        og.sim.render()
    return env.get_observation()


def _to_numpy_array(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_plain_scalar(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def _find_seg_key_for_object(seg_info, object_name):
    if not isinstance(seg_info, dict):
        return None
    for key, value in seg_info.items():
        if value == object_name or str(value) == object_name:
            return _to_plain_scalar(key)
    for key, value in seg_info.items():
        if object_name in str(value):
            return _to_plain_scalar(key)
    return None


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def _project_world_point_to_camera(point_world, sensor):
    try:
        sensor_pos, sensor_quat = sensor.get_position_orientation()
        camera_mat = T.pose2mat((sensor_pos, sensor_quat))
        target_h = th.ones(4, dtype=th.float32)
        target_h[:3] = th.as_tensor(point_world, dtype=th.float32)
        target_cam = th.linalg.inv(camera_mat).to(dtype=th.float32) @ target_h
        target_cam = th.stack([target_cam[0], target_cam[1], -target_cam[2], target_cam[3]])
        z = float(target_cam[2].item())
        if z <= 1e-4:
            return {"in_front": False, "in_frame": False, "reason": "behind_camera"}
        focal_length = float(getattr(sensor, "focal_length", 17.0) or 17.0)
        horizontal_aperture = float(getattr(sensor, "horizontal_aperture", 20.995) or 20.995)
        image_width = int(getattr(sensor, "image_width", 1) or 1)
        image_height = int(getattr(sensor, "image_height", 1) or 1)
        fx = focal_length * image_width / horizontal_aperture
        fy = fx
        cx = image_width / 2.0
        cy = image_height / 2.0
        u = fx * float(target_cam[0].item()) / z + cx
        v = fy * float(target_cam[1].item()) / z + cy
        return {
            "in_front": True,
            "in_frame": bool(0 <= u < image_width and 0 <= v < image_height),
            "pixel": [float(u), float(v)],
            "depth": z,
            "image_width": image_width,
            "image_height": image_height,
        }
    except Exception as e:
        return {"in_front": False, "in_frame": False, "error": f"{type(e).__name__}: {e}"}


def _obs_visibility_for_object(env, obs, obs_info, object_name, marker_pos=None):
    og_env = getattr(env, "env", env)
    robot = og_env.robots[0] if getattr(og_env, "robots", None) else None
    robot_name = getattr(robot, "name", "robot_r1") if robot is not None else "robot_r1"
    out = {}
    for cam_name, sensor_name in R1PRO_OBS_CAMERA_SENSOR_NAMES.items():
        seg_key = f"{robot_name}::{sensor_name}::seg_instance"
        rgb_key = f"{robot_name}::{sensor_name}::rgb"
        seg = _to_numpy_array(obs.get(seg_key))
        rgb = _to_numpy_array(obs.get(rgb_key))
        sensor_info = (((obs_info or {}).get(robot_name) or {}).get(sensor_name) or {})
        obj_seg_key = _find_seg_key_for_object(sensor_info.get("seg_instance"), object_name)
        row = {
            "seg_key": seg_key,
            "rgb_key": rgb_key,
            "object_segmentation_key": obj_seg_key,
            "object_visible": False,
            "object_pixel_count": 0,
            "object_pixel_fraction": 0.0,
            "object_bbox": None,
            "object_bbox_area_fraction": 0.0,
        }
        if seg is None:
            row["error"] = "seg_instance_missing"
        elif obj_seg_key is None:
            row["error"] = "object_not_in_seg_instance_info"
        else:
            mask = seg == obj_seg_key
            pixel_count = int(mask.sum())
            total = int(mask.size)
            bbox = _bbox_from_mask(mask)
            row.update({
                "object_visible": pixel_count > 0,
                "object_pixel_count": pixel_count,
                "object_pixel_fraction": (pixel_count / total) if total else 0.0,
                "object_bbox": bbox,
            })
            if bbox is not None and total:
                row["object_bbox_area_fraction"] = (bbox["width"] * bbox["height"]) / total
        if rgb is not None:
            row["rgb_shape"] = list(rgb.shape)
        if marker_pos is not None and robot is not None:
            sensor = getattr(robot, "sensors", {}).get(sensor_name)
            row["marker_projection"] = (
                _project_world_point_to_camera(marker_pos, sensor)
                if sensor is not None else {"in_front": False, "in_frame": False, "error": "sensor_missing"}
            )
        out[cam_name] = row
    return out


def _capture_obs_layout_frame(env):
    """Capture the model-observation camera layout used for semantic review.

    Layout:
      - left column: left wrist over right wrist, each 224x224
      - right column: head camera, 448x448
    """
    obs, _ = _get_obs(env)
    missing = []
    frames = {}
    for name, key in R1PRO_OBS_CAMERA_KEYS.items():
        obs_key = _find_obs_key(obs, key)
        frame = _to_numpy_rgb(obs.get(obs_key) if obs_key is not None else None)
        if frame is None:
            missing.append(key)
        else:
            frames[name] = frame
    if missing:
        available_rgb = sorted(str(k) for k in obs.keys() if "rgb" in str(k).lower())
        raise RuntimeError(
            "Cannot build obs layout video; missing camera observations "
            f"{missing}. Available rgb keys: {available_rgb[:20]}"
        )
    left = cv2.resize(frames["left_wrist"], (224, 224), interpolation=cv2.INTER_AREA)
    right = cv2.resize(frames["right_wrist"], (224, 224), interpolation=cv2.INTER_AREA)
    head = cv2.resize(frames["head"], (448, 448), interpolation=cv2.INTER_AREA)
    return np.hstack([np.vstack([left, right]), head]).copy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--snapshot-every', type=int, default=1)
    ap.add_argument('--init-curobo', action='store_true')
    ap.add_argument('--call-wrapper-reset', action='store_true', help='Call EnvOmniGibson.reset() before loading states[0]. Disabled by default because replay restores the exact HDF5 initial state.')
    ap.add_argument('--target-object', action='append', default=None, help='Only record diagnostics for this object name. Can be repeated.')
    ap.add_argument('--video-output', default=None, help='Optional MP4 path for rendered replay video.')
    ap.add_argument('--video-every', type=int, default=1, help='Append one video frame every N replay steps.')
    ap.add_argument('--video-fps', type=int, default=20)
    ap.add_argument('--obs-video-output', default=None, help='Optional MP4 path for observation-camera layout replay video.')
    ap.add_argument('--obs-video-every', type=int, default=None, help='Append one observation-layout frame every N replay steps. Defaults to --video-every.')
    ap.add_argument('--obs-video-fps', type=int, default=None, help='Observation-layout video FPS. Defaults to --video-fps.')
    ap.add_argument('--obs-visibility', action='store_true', help='Record per-camera object segmentation and marker projection visibility metrics.')
    args=ap.parse_args()
    status_path=f"{args.output}.status.json"
    status={'dataset':args.dataset,'output':args.output,'stage':'starting'}
    write_payload(status_path,status)
    gm.ENABLE_TRANSITION_RULES=False
    with h5py.File(args.dataset,'r') as f:
        status['stage']='reading_hdf5'
        write_payload(status_path,status)
        env_meta=json.loads(f['data'].attrs['env_args'])
        grp=f['data/demo_0']
        actions=np.asarray(grp['actions'])
        states=np.asarray(grp['states'])
        target_objects=args.target_object
        if target_objects is None and 'datagen_info/object_poses' in grp:
            target_objects=list(grp['datagen_info/object_poses'].keys())
        total=len(actions)
    end=total if args.end<0 else min(args.end,total)
    status.update({'stage':'creating_env','total_actions':int(total),'start':int(args.start),'end':int(end)})
    write_payload(status_path,status)
    env=RobomimicUtils.create_env(env_meta, init_curobo=args.init_curobo)
    _ensure_camera_modalities(env, include_visibility=args.obs_visibility)
    status['stage']='env_created'
    write_payload(status_path,status)
    records=[]
    video_writer=None
    obs_video_writer=None
    obs_video_frames=0
    obs_video_every=max(1, args.obs_video_every if args.obs_video_every is not None else args.video_every)
    obs_video_fps=args.obs_video_fps if args.obs_video_fps is not None else args.video_fps
    try:
        # Reset then restore exact generated initial state.
        if args.call_wrapper_reset:
            status['stage']='resetting'
            write_payload(status_path,status)
            env.reset()
        status['stage']='loading_initial_state'
        write_payload(status_path,status)
        og.sim.load_state(th.as_tensor(states[args.start]), serialized=True)
        if args.video_output:
            Path(args.video_output).parent.mkdir(parents=True, exist_ok=True)
            video_writer=imageio.get_writer(args.video_output, fps=args.video_fps)
            # Let the restored state propagate into the viewer camera before the first frame.
            for _ in range(3):
                og.sim.render()
            video_writer.append_data(_capture_frame(env, target_objects=target_objects))
        if args.obs_video_output:
            Path(args.obs_video_output).parent.mkdir(parents=True, exist_ok=True)
            obs_video_writer=imageio.get_writer(args.obs_video_output, fps=obs_video_fps)
            obs_video_writer.append_data(_capture_obs_layout_frame(env))
            obs_video_frames += 1
        for i in range(args.start, end):
            status.update({'stage':'replaying','current_step':int(i),'num_records':len(records)})
            write_payload(status_path,status)
            if i == args.start or ((i-args.start) % max(1,args.snapshot_every)==0):
                record = _snapshot(env, i, ref_state=states[i], target_objects=target_objects)
                if args.obs_visibility:
                    obs, obs_info = _get_obs(env)
                    for obj_name, row in record.get("objects", {}).items():
                        row["observation_visibility"] = _obs_visibility_for_object(
                            env,
                            obs,
                            obs_info,
                            obj_name,
                            marker_pos=row.get("marker_pos"),
                        )
                records.append(record)
            env.step(actions[i])
            if video_writer is not None and ((i - args.start + 1) % max(1,args.video_every)==0):
                video_writer.append_data(_capture_frame(env, target_objects=target_objects))
            if obs_video_writer is not None and ((i - args.start + 1) % obs_video_every==0):
                obs_video_writer.append_data(_capture_obs_layout_frame(env))
                obs_video_frames += 1
        record = _snapshot(env, end, ref_state=states[end] if end < len(states) else None, target_objects=target_objects)
        if args.obs_visibility:
            obs, obs_info = _get_obs(env)
            for obj_name, row in record.get("objects", {}).items():
                row["observation_visibility"] = _obs_visibility_for_object(
                    env,
                    obs,
                    obs_info,
                    obj_name,
                    marker_pos=row.get("marker_pos"),
                )
        records.append(record)
        if video_writer is not None:
            video_writer.close()
            video_writer=None
        if obs_video_writer is not None:
            obs_video_writer.close()
            obs_video_writer=None
        payload={
            'dataset':args.dataset,
            'start':args.start,
            'end':end,
            'video_output':args.video_output,
            'obs_video_output':args.obs_video_output,
            'obs_video_frames':obs_video_frames,
            'summary':summarize(records),
            'records':records,
        }
        write_payload(args.output,payload)
        status.update({'stage':'completed','num_records':len(records),'summary':payload['summary']})
        write_payload(status_path,status)
        print(json.dumps(payload['summary'],indent=2))
        print(args.output)
        sys.stdout.flush()
    except BaseException as e:
        payload={
            'dataset':args.dataset,
            'start':args.start,
            'end':end,
            'error':f'{type(e).__name__}: {e}',
            'traceback':traceback.format_exc(),
            'records':records,
            'summary':summarize(records),
        }
        write_payload(args.output,payload)
        status.update({'stage':'failed','error':payload['error'],'traceback':payload['traceback'],'num_records':len(records),'summary':payload['summary']})
        write_payload(status_path,status)
        print(json.dumps({'error':payload['error'],'summary':payload['summary']},indent=2), file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
        if obs_video_writer is not None:
            try:
                obs_video_writer.close()
            except Exception:
                pass
        try:
            status['stage']='closing_env'
            write_payload(status_path,status)
            env.close()
        except Exception:
            pass
        try:
            status['stage']='shutdown'
            write_payload(status_path,status)
        except Exception:
            pass
        og.shutdown()

if __name__=='__main__':
    main()
