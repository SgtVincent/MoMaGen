"""
MoMaGen environment interface classes for OmniGibson environments.
Refactored to use configuration-driven tasks instead of hardcoded classes.
"""
import json
import os
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field, replace
import cvxpy as cp
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import *
from omnigibson.controllers import ControlType

from momagen.env_interfaces.base import MG_EnvInterface
from momagen.datagen.datagen_info import DatagenInfo


@dataclass
class TaskConfig:
    """Configuration for a task, defining objects and termination signals."""
    name: str
    tracked_objects: Dict[str, str] = field(default_factory=dict)  # logical_name -> registry_name
    termination_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # signal_name -> config
    robot_specific_objects: Dict[str, Dict[str, str]] = field(default_factory=dict)  # robot_type -> object_mapping
    bimanual: bool = True  # Whether this task uses bimanual interface


class OmniGibsonInterface(MG_EnvInterface):
    """
    MoMaGen environment interface base class for basic omnigibson environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "omnigibson"

    def __init__(self, env, task_config: TaskConfig = None):
        super(OmniGibsonInterface, self).__init__(env)
        self.task_config = task_config
        self.robot = env.robots[0]
        self._setup_arm_controller()

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.
        """
        start_idx = 0
        end_idx = None
        arm_controller = None
        for controller_type, controller in self.robot.controllers.items():
            if controller_type != f"arm_{self.robot.default_arm}":
                start_idx += controller.command_dim
            else:
                end_idx = start_idx + controller.command_dim
                arm_controller = controller
                break
        assert end_idx is not None and arm_controller is not None
        self.arm_controller = arm_controller

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        return self.get_object_pose(self.robot.eef_links[self.robot.default_arm])

    def get_object_pose(self, obj):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj (BaseObject): OG object

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        return T.pose2mat(obj.get_position_orientation())

    def _get_object_by_name(self, name: str):
        """Get object from scene by name, with fallback strategies."""
        try:
            # Try object scope first (for newer tasks)
            if hasattr(self.env.task, 'object_scope') and name in self.env.task.object_scope:
                return self.env.task.object_scope[name]

            # Try scene registry
            return self.env.scene.object_registry("name", name)
        except (KeyError, AttributeError):
            # If object not found, return None (caller should handle)
            return None

    def get_object_poses(self):
        """Gets poses of task-relevant objects based on configuration."""
        if not self.task_config:
            return {}

        object_poses = {}

        # Get robot-specific objects if applicable
        robot_type = type(self.robot).__name__.lower()
        if robot_type in self.task_config.robot_specific_objects:
            tracked_objects = {**self.task_config.tracked_objects,
                             **self.task_config.robot_specific_objects[robot_type]}
        else:
            tracked_objects = self.task_config.tracked_objects

        for logical_name, registry_name in tracked_objects.items():
            # Special handling for robot links (e.g., torso_link4)
            if logical_name == "torso_link4" and registry_name in ["torso_lift_link", "torso_link4"]:
                # This is a robot link, not a scene object
                robot_link = self.env.robots[0].links[registry_name]
                object_poses[logical_name] = self.get_object_pose(robot_link)
            else:
                obj = self._get_object_by_name(registry_name)
                if obj is not None:
                    object_poses[logical_name] = self.get_object_pose(obj)

        return object_poses

    def get_subtask_term_signals(self):
        """Gets termination signals based on configuration."""
        if not self.task_config:
            return {}

        signals = {}

        for signal_name, signal_config in self.task_config.termination_signals.items():
            signal_type = signal_config.get("type")
            obj_name = signal_config.get("object")

            obj = self._get_object_by_name(obj_name) if obj_name else None

            if signal_type == "grasp" and obj:
                arm = signal_config.get("arm", "default")
                signals[signal_name] = int(self.robot.is_grasping(arm=arm, candidate_obj=obj))

            elif signal_type == "touch" and obj:
                signals[signal_name] = int(self.robot.states[Touching].get_value(obj))

            elif signal_type == "custom":
                # Allow custom signal evaluation via callback
                callback = signal_config.get("callback")
                if callable(callback):
                    signals[signal_name] = callback(self.env, self.robot, obj)

        return signals

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller (in the world frame) and returns an action
        (usually a normalized delta pose action in the robot frame) to try and achieve that target pose.

        Args:
            target_pose (np.array): 4x4 target eef pose, in the world frame

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation), in the robot frame
        """
        # Legacy
        del relative

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)

        # Compute the eef target pose in the robot frame
        target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
        dpos = target_pos - pos_relative

        dori = T.orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))

        # Assemble the arm command and undo the preprocessing
        arm_command = th.cat([dpos, dori])
        arm_action = self.robot.controllers[f"arm_{self.robot.default_arm}"]._reverse_preprocess_command(arm_command)

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = th.from_numpy(np.zeros_like(self.robot.action_space.sample())[:-1])
        action[self.robot.arm_action_idx[self.robot.default_arm]] = arm_action

        # Convert to numpy tensor
        action = action.numpy()

        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        # Legacy
        del relative

        # Ensure float32
        action = action.astype(np.float32)

        # Convert to torch tensor
        action = th.from_numpy(action)

        # Extract the arm command part of the action and preprocess it
        arm_action = action[self.robot.arm_action_idx[self.robot.default_arm]]
        arm_command = self.robot.controllers[f"arm_{self.robot.default_arm}"]._preprocess_command(arm_action)

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Extract the delta pose from the arm command and compute the target pose in the robot frame
        dpos = arm_command[:3]
        target_pos = pos_relative + dpos
        dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
        target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Convert the target pose to the world frame
        target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
        target_pose = target_pose.numpy()

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action[:-1], th.from_numpy(new_action), atol=1e-2)

        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        # last dimension is gripper action
        return action[-1:]


class OmniGibsonInterfaceBimanual(OmniGibsonInterface):
    """
    MimicGen environment interface class for bimanual robots.
    """
    INTERFACE_TYPE = "omnigibson_bimanual"

    def __init__(self, env, task_config: TaskConfig = None):
        super(OmniGibsonInterfaceBimanual, self).__init__(env, task_config)
        self._setup_arm_controller()
        self.gripper_action_dim = th.cat([self.robot.controller_action_idx["gripper_left"], self.robot.controller_action_idx["gripper_right"]])
        if os.environ.get("MOMAGEN_DEBUG_GRIPPER") == "1":
            print(
                "[MOMAGEN_DEBUG_GRIPPER] "
                f"robot_action_dim={self.robot.action_dim} "
                f"controller_order={getattr(self.robot, 'controller_order', None)} "
                f"controller_action_idx={self.robot.controller_action_idx} "
                f"gripper_action_dim={self.gripper_action_dim.tolist()}"
            )

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.
        """
        self.robot = self.env.robots[0]
        self.arm_controller = {}

    def get_robot_eef_pose(self, arm_name=None):
        """
        Get current robot end effector pose for specified arm or both.

        Returns:
            pose (np.array): 4x4 eef pose matrix for single arm, or 8x4 for both arms
        """
        if arm_name:
            assert arm_name in ["left", "right"]
            return self.get_object_pose(self.robot.eef_links[arm_name])
        else:
            # Return concatenated poses for both arms
            left_pose = self.get_object_pose(self.robot.eef_links["left"])
            right_pose = self.get_object_pose(self.robot.eef_links["right"])
            return np.concatenate([left_pose, right_pose], axis=0)

    def get_datagen_info(self, action=None):
        """
        Get information needed for data generation, at the current
        timestep of simulation. If @action is provided, it will be used to
        compute the target eef pose for the controller, otherwise that
        will be excluded.

        Returns:
            datagen_info (DatagenInfo instance)
        """

        # current eef pose
        eef_pose = self.get_robot_eef_pose()  # 8x4 for bimanual
        base_pose = self.get_object_pose(self.robot)

        # object poses
        object_poses = self.get_object_poses()

        # subtask termination signals
        subtask_term_signals = self.get_subtask_term_signals()

        # these must be extracted from provided action
        # Only record eef_pose that are actually achieved, not the target_pose
        gripper_action = None
        if action is not None:
            gripper_action = self.action_to_gripper_action(action)

        datagen_info = DatagenInfo(
            base_pose=base_pose,
            eef_pose=eef_pose,
            object_poses=object_poses,
            subtask_term_signals=subtask_term_signals,
            gripper_action=gripper_action,
        )
        return datagen_info

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        gripper_action = action[self.gripper_action_dim]
        return gripper_action

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller (in the world frame) and returns an action
        (usually a normalized delta pose action in the robot frame) to try and achieve that target pose.

        Args:
            target_pose (np.array): 4x4 target eef pose, in the world frame

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation), in the robot frame
        """
        # Legacy
        del relative

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)
        target_pose_dict = {}
        target_pose_dict["left"] = target_pose[:4,:]
        target_pose_dict["right"] = target_pose[4:,:]

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = np.zeros_like(self.robot.action_space.sample())
        
        control_dict = self.robot.get_control_dict()

        trunk_controller = self.robot.controllers.get("trunk")
        include_trunk_arms_raw = os.environ.get("MOMAGEN_IK_INCLUDE_TRUNK_ARMS", "")
        include_trunk_arms = {
            item.strip().lower()
            for item in include_trunk_arms_raw.split(",")
            if item.strip()
        }
        include_trunk_for_all = os.environ.get("MOMAGEN_IK_INCLUDE_TRUNK", "0") == "1"
        trunk_action_written = False

        for arm_name in ["left", "right"]:
            target_pose = target_pose_dict[arm_name]

            # Compute the eef target pose in the robot frame
            target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
            dpos = target_pos - pos_relative

            dori = T.orientation_error(T.quat2mat(target_quat), T.quat2mat(quat_relative))

            # Compute delta pose
            err = th.cat([dpos, dori])

            # Replicate the logic from IKController
            arm_controller = self.robot.controllers[f"arm_{arm_name}"]
            arm_dof_idx = arm_controller.dof_idx
            manipulation_dof_idx = arm_dof_idx

            include_trunk = (
                trunk_controller is not None
                and (
                    include_trunk_for_all
                    or arm_name in include_trunk_arms
                    or "both" in include_trunk_arms
                    or "all" in include_trunk_arms
                )
            )
            trunk_dof_idx = None
            if include_trunk:
                trunk_dof_idx = trunk_controller.dof_idx
                manipulation_dof_idx = np.concatenate([
                    np.asarray(arm_dof_idx, dtype=int),
                    np.asarray(trunk_dof_idx, dtype=int),
                ])

            j_eef = control_dict[f"eef_{arm_name}_jacobian_relative"][:, manipulation_dof_idx]
            q = control_dict["joint_position"][manipulation_dof_idx]
            q_lower_limit = arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx]
            q_upper_limit = arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx]

            # percentile = 0.95
            # q_range = q_upper_limit - q_lower_limit
            # q_lower_limit = q_lower_limit + (1 - percentile) / 2 * q_range
            # q_upper_limit = q_upper_limit - (1 - percentile) / 2 * q_range
            # q = np.clip(q, q_lower_limit, q_upper_limit)

            q_dot_lower_limit = arm_controller._control_limits[ControlType.get_type("velocity")][0][manipulation_dof_idx]
            q_dot_upper_limit = arm_controller._control_limits[ControlType.get_type("velocity")][1][manipulation_dof_idx]

            vel_err = err.numpy() / og.sim.get_physics_dt()
            proportional_gain = float(os.environ.get("MOMAGEN_IK_PROPORTIONAL_GAIN", "0.5") or 0.5)

            n = j_eef.shape[1]
            epsilon = 1e-6
            P = j_eef.T @ j_eef + epsilon * np.eye(j_eef.shape[1])
            r = -proportional_gain * vel_err @ j_eef

            velocity_gain = float(os.environ.get("MOMAGEN_IK_VELOCITY_GAIN", "0.5") or 0.5)
            q_dot_upper_limit_by_joint_limit = velocity_gain * (q_upper_limit - q) / og.sim.get_physics_dt()
            q_dot_lower_limit_by_joint_limit = velocity_gain * (q_lower_limit - q) / og.sim.get_physics_dt()

            q_dot_upper_limit = np.minimum(q_dot_upper_limit, q_dot_upper_limit_by_joint_limit)
            q_dot_lower_limit = np.maximum(q_dot_lower_limit, q_dot_lower_limit_by_joint_limit)

            G = np.vstack([np.eye(n), -np.eye(n)])
            h = np.concatenate([q_dot_upper_limit, -q_dot_lower_limit])

            q_dot = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(q_dot, P) + r.T @ q_dot), [G @ q_dot <= h])
            prob_status = None
            q_dot_val = None
            unclipped_target_joint_pos = None
            try:
                prob.solve()
            except cp.error.SolverError:
                target_joint_pos = q
                prob_status = "solver_error"
            else:
                prob_status = prob.status
                if prob.status == "optimal":
                    q_dot_val = q_dot.value
                    delta_j = q_dot_val * og.sim.get_physics_dt()
                    target_joint_pos = q + delta_j
                else:
                    target_joint_pos = q

            # NOTE: This clipping is important because the convex optimization (cvxpy) solver does not guarantee that the solution will be STRICTLY within the limits
            # The result is that sometimes the joint positions obtained from the solver are just slightly (even in the order of 1e-5) out of the limits
            # So, making the limits of target_joint_pos (in radians) a bit more stricter will help avoid this issue
            unclipped_target_joint_pos = target_joint_pos
            joint_limit_margin = float(os.environ.get("MOMAGEN_IK_JOINT_LIMIT_MARGIN", "0.02") or 0.02)
            target_joint_pos = np.clip(target_joint_pos, q_lower_limit + joint_limit_margin, q_upper_limit - joint_limit_margin)

            arm_command = target_joint_pos
            trunk_action = None
            if include_trunk:
                arm_dof_count = int(arm_dof_idx.shape[0])
                arm_command = target_joint_pos[:arm_dof_count]
                trunk_command = target_joint_pos[arm_dof_count:]
                trunk_action = trunk_controller._reverse_preprocess_command(trunk_command)
                action[self.robot.controller_action_idx["trunk"]] = trunk_action
                trunk_action_written = True

            arm_action = arm_controller._reverse_preprocess_command(arm_command)
            action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_action

            debug_ik_action = os.environ.get("MOMAGEN_DEBUG_IK_ACTION") == "1"
            if debug_ik_action:
                debug_ik_every = int(os.environ.get("MOMAGEN_DEBUG_IK_ACTION_EVERY", "1") or 1)
                self._momagen_debug_ik_action_counter = getattr(self, "_momagen_debug_ik_action_counter", 0) + 1

                def _as_np(x):
                    if x is None:
                        return None
                    if hasattr(x, "detach"):
                        x = x.detach()
                    if hasattr(x, "cpu"):
                        x = x.cpu()
                    return np.asarray(x, dtype=float)

                q_np = _as_np(q)
                target_np = _as_np(target_joint_pos)
                unclipped_np = _as_np(unclipped_target_joint_pos)
                q_dot_np = _as_np(q_dot_val)
                arm_action_np = _as_np(arm_action)
                trunk_action_np = _as_np(trunk_action)
                q_lower_np = _as_np(q_lower_limit)
                q_upper_np = _as_np(q_upper_limit)
                clip_delta_norm = None
                clipped_joint_count = None
                joint_margin_min = None
                if target_np is not None and unclipped_np is not None:
                    clip_delta = target_np - unclipped_np
                    clip_delta_norm = float(np.linalg.norm(clip_delta))
                    clipped_joint_count = int(np.sum(np.abs(clip_delta) > 1e-6))
                if target_np is not None and q_lower_np is not None and q_upper_np is not None:
                    joint_margin_min = float(np.min(np.minimum(target_np - q_lower_np, q_upper_np - target_np)))
                target_joint_delta_norm = (
                    float(np.linalg.norm(target_np - q_np))
                    if target_np is not None and q_np is not None
                    else None
                )
                should_print_ik_debug = (
                    debug_ik_every <= 1
                    or self._momagen_debug_ik_action_counter % debug_ik_every == 0
                    or prob_status not in {"optimal", "optimal_inaccurate"}
                    or (clipped_joint_count is not None and clipped_joint_count > 0)
                )
                if should_print_ik_debug:
                    print(
                        "[MOMAGEN_DEBUG_IK_ACTION] "
                        f"arm={arm_name} dpos_norm={float(th.linalg.norm(dpos)):.6f} "
                        f"dori_norm={float(th.linalg.norm(dori)):.6f} include_trunk={include_trunk} "
                        f"status={prob_status} "
                        f"qdot_norm={float(np.linalg.norm(q_dot_np)) if q_dot_np is not None else None} "
                        f"target_joint_delta_norm={target_joint_delta_norm} "
                        f"action_norm={float(np.linalg.norm(arm_action_np)) if arm_action_np is not None else None} "
                        f"trunk_action_norm={float(np.linalg.norm(trunk_action_np)) if trunk_action_np is not None else None} "
                        f"clip_delta_norm={clip_delta_norm} clipped_joint_count={clipped_joint_count} "
                        f"joint_margin_min={joint_margin_min}",
                        flush=True,
                    )

        # fill in the no operation actions for the base, camera and trunk
        for name, controller in self.robot.controllers.items():
            if name == "trunk" and trunk_action_written:
                continue
            if name == 'base' or name == 'camera' or name == "trunk":
                partial_action = controller.compute_no_op_action(control_dict)
                action_idx = self.robot.controller_action_idx[name]
                action[action_idx] = partial_action

        return action

    def generate_action(self, target_pose):
        """
        Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

        Returns:
            th.tensor or None: Action array for one step for the robot to do nothing
        """
        # change to quaternion 

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)
        target_pose_dict = {}
        target_pose_dict["left"] = T.mat2pose(target_pose[:4,:]) # T.mat2pose(target_pose)
        target_pose_dict["right"] = T.mat2pose(target_pose[4:,:])
        
        arm_targets = {
            'arm_left': (target_pose_dict["left"][0], target_pose_dict["left"][1], 0),
            'arm_right': (target_pose_dict["right"][0], target_pose_dict["right"][1], 0),
        }

        action = th.zeros(self.robot.action_dim)
        for name, controller in self.robot.controllers.items():
            # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
            if name in arm_targets:
                arm = name.replace("arm_", "")
                # change to robot base frame
                target_pos, target_orn, gripper_state = arm_targets[name] # in world fixed frame

                current_pos, current_orn = self.robot.get_eef_pose(arm)
                if target_orn is None:
                    target_orn = current_orn
                if target_pos is None:
                    target_pos = current_pos
                arm_targets[name] = (target_pos, target_orn, gripper_state)

                delta_pos = target_pos - current_pos
                delta_orn = T.orientation_error(T.quat2mat(target_orn), T.quat2mat(current_orn))
                partial_action = th.cat((delta_pos, delta_orn))
            else:
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
            action_idx = self.robot.controller_action_idx[name]
            action[action_idx] = partial_action

            # set the gripper no operation action to 0
            action[11] = 0
            action[-1] = 0
        
        # bug: change to robot base frame

        # Convert to numpy tensor
        action = action.numpy()
        print('generated action')
        print('arm left')
        print(action[5:12])
        print('arm right')
        print(action[12:19])
        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        # Legacy
        del relative

        # Ensure float32
        action = action.astype(np.float32)

        # Convert to torch tensor
        action = th.from_numpy(action)

        target_pose_dict = {}

        for arm_name in ["left", "right"]:
            # Extract the arm command part of the action and preprocess it
            arm_action = action[self.robot.arm_action_idx[arm_name]]
            arm_command = self.robot.controllers[f"arm_{arm_name}"]._preprocess_command(arm_action)

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Extract the delta pose from the arm command and compute the target pose in the robot frame
            dpos = arm_command[:3]
            target_pos = pos_relative + dpos
            dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

            # Convert the target pose to the world frame
            target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
            target_pose = target_pose.numpy()
            
            target_pose_dict[arm_name] = target_pose

        target_pose = np.concatenate([target_pose_dict["left"], target_pose_dict["right"]], axis=0) # 8x4

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        new_action[self.gripper_action_dim] = action[self.gripper_action_dim]
        new_action[:5] = action[:5]

        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action, th.from_numpy(new_action), atol=1e-2)


        return target_pose


# Task configuration definitions
TASK_CONFIGS = {

    "test_tiago_cup": TaskConfig(
        name="test_tiago_cup",
        tracked_objects={
            "coffee_cup": "coffee_cup",
            "teacup": "teacup",
            "breakfast_table": "breakfast_table",
        },
        termination_signals={
            "grasp_right": {
                "type": "grasp",
                "object": "coffee_cup",
                "arm": "right"
            }
        }
    ),

    "test_r1_cup": TaskConfig(
        name="test_r1_cup",
        tracked_objects={
            "coffee_cup": "coffee_cup",
            "teacup": "teacup",
            "breakfast_table": "breakfast_table",
        },
        termination_signals={
            "grasp_right": {
                "type": "grasp",
                "object": "coffee_cup",
                "arm": "right"
            }
        }
    ),

    "r1_tidy_table": TaskConfig(
        name="r1_tidy_table",
        tracked_objects={
            "teacup_601": "teacup_601",
            "drop_in_sink_awvzkn_0": "drop_in_sink_awvzkn_0",
        },
    ),

    "r1_pick_cup": TaskConfig(
        name="r1_pick_cup",
        tracked_objects={
            "coffee_cup_7": "coffee_cup_7",
            "breakfast_table_6": "breakfast_table_6",
        },
    ),

    "r1_dishes_away": TaskConfig(
        name="r1_dishes_away",
        tracked_objects={
            "countertop_kelker_0": "countertop_kelker_0",
            "shelf_pfusrd_1": "shelf_pfusrd_1",
            "plate_603": "plate_603",
            "plate_602": "plate_602",
            "plate_601": "plate_601",
        },
    ),

    "r1_clean_pan": TaskConfig(
        name="r1_clean_pan",
        tracked_objects={
            "frying_pan_602": "frying_pan_602",
            "scrub_brush_601": "scrub_brush_601",
            "robot_r1": "robot_r1",
        },
        robot_specific_objects={
            "tiago": {"torso_link4": "torso_lift_link"},
            "r1": {"torso_link4": "torso_link4"},
        },
    ),

    "r1_bringing_water": TaskConfig(
        name="r1_bringing_water",
        tracked_objects={
            "beer_bottle_595": "beer_bottle_595",
            "fridge_dszchb_0": "fridge_dszchb_0",
        },
    ),

    "r1_picking_up_trash": TaskConfig(
        name="r1_picking_up_trash",
        tracked_objects={
            "can_of_soda_114": "can_of_soda_114",
            "trash_can_116": "trash_can_116",
            "can_of_soda_261": "can_of_soda_261",
            "trash_can_262": "trash_can_262",
        },
    ),

    "r1_turning_on_radio": TaskConfig(
        name="r1_turning_on_radio",
        tracked_objects={
            "radio_89": "radio_89",
            "coffee_table_koagbh_0": "coffee_table_koagbh_0",
        },
    ),

    # ------------------------------------------------------------------------------------------------
    # Add new task configs here
    # Note 1: tracked_objects is a dictionary with the same key and value. Furthermore, the tracked_object is the
    # OG specific name of the object which you can find by clicking on the object on the GUI
    # Note 2: we are currently not using the termination_signals for the data generation, so you can leave it empty
    # ------------------------------------------------------------------------------------------------
   
}


def _get_task_config(task_name):
    """Return task config, optionally applying local experiment overrides.

    Set ``MOMAGEN_TASK_CONFIG_OVERRIDES`` to a JSON object like:
    ``{"r1_picking_up_trash": {"tracked_objects": {"can": "can_1"}}}``.
    This keeps local BEHAVIOR source-demo object ids out of the checked-in base
    configs while making one-off source-demo conversion possible.
    """
    task_config = TASK_CONFIGS[task_name]
    overrides_json = os.environ.get("MOMAGEN_TASK_CONFIG_OVERRIDES")
    if not overrides_json:
        return task_config

    overrides = json.loads(overrides_json)
    task_overrides = overrides.get(task_name, {})
    if not task_overrides:
        return task_config

    return replace(
        task_config,
        tracked_objects=task_overrides.get("tracked_objects", task_config.tracked_objects),
        termination_signals=task_overrides.get("termination_signals", task_config.termination_signals),
        robot_specific_objects=task_overrides.get("robot_specific_objects", task_config.robot_specific_objects),
        bimanual=task_overrides.get("bimanual", task_config.bimanual),
    )

# Backward compatibility - create legacy classes that use the new system
class MG_TestTiagoCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("test_tiago_cup"))

class MG_TestR1Cup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("test_r1_cup"))

class MG_R1PutAwayCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_put_away_cup"))

class MG_R1TidyTable(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_tidy_table"))

class MG_R1PickCup(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_pick_cup"))

class MG_R1DishesAway(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_dishes_away"))

class MG_R1CleanPan(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_clean_pan"))

class MG_R1BringingWater(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_bringing_water"))

# ---------------------------------
# Add new class here for new tasks
# ---------------------------------

class MG_R1PickingUpTrash(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_picking_up_trash"))

class MG_R1TurningOnRadio(OmniGibsonInterfaceBimanual):
    def __init__(self, env):
        super().__init__(env, _get_task_config("r1_turning_on_radio"))
