"""
Collection of utilities related to robomimic.
"""
import json
import traceback
import argparse
import importlib.util
import os
from copy import deepcopy

import robomimic
import robomimic.utils.env_utils as EnvUtils
from robomimic.scripts.playback_dataset import playback_dataset, DEFAULT_CAMERAS




def create_env(
    env_meta,
    env_name=None,
    env_class=None,
    robot=None,
    gripper=None,
    camera_names=None,
    camera_height=84,
    camera_width=84,
    render=None, 
    render_offscreen=None, 
    use_image_obs=None, 
    use_depth_obs=None, 
    init_curobo=True,
    policy_rollout=False,
    manipulation_only=False,
    real_robot_mode=False,
    baseline=None
):
    """
    Helper function to create the environment from dataset metadata and arguments.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html
        env_name (str or None): if provided, override environment name 
            in @env_meta
        env_class (class or None): if provided, use this class instead of the
            one inferred from @env_meta
        robot (str or None): if provided, override the robot argument in
            @env_meta. Currently only supported by robosuite environments.
        gripper (str or None): if provided, override the gripper argument in
            @env_meta. Currently only supported by robosuite environments.
        camera_names (list of str or None): list of camera names that correspond to image observations
        camera_height (int): camera height for all cameras
        camera_width (int): camera width for all cameras
        render (bool or None): optionally override rendering behavior
        render_offscreen (bool or None): optionally override rendering behavior
        use_image_obs (bool or None): optionally override rendering behavior
        use_depth_obs (bool or None): optionally override rendering behavior
    """
    env_meta = deepcopy(env_meta)

    # maybe override some settings in environment metadata
    if env_name is not None:
        env_meta["env_name"] = env_name
    if robot is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert robot in ["IIWA", "Sawyer", "UR5e", "Panda", "Jaco", "Kinova3"]
        env_meta["env_kwargs"]["robots"] = [robot]
    if gripper is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert gripper in ["PandaGripper", "RethinkGripper", "Robotiq85Gripper", "Robotiq140Gripper"]
        env_meta["env_kwargs"]["gripper_types"] = [gripper]

    if camera_names is None:
        camera_names = []

    # create environment
    env_type = EnvUtils.get_env_type(env_meta=env_meta)
    if env_type == 4:
        # The current behavior env ships an older robomimic whose
        # create_env_for_data_processing signature cannot accept custom env
        # classes / OG-specific kwargs, and whose env registry does not know
        # EnvType.OG_TYPE. Load MoMaGen's local EnvOmniGibson wrapper directly.
        import robomimic.envs.env_base as EB
        if not hasattr(EB.EnvType, "OG_TYPE"):
            EB.EnvType.OG_TYPE = 4

        env_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "robomimic", "robomimic", "envs", "env_omnigibson.py"
        ))
        spec = importlib.util.spec_from_file_location("momagen_local_env_omnigibson", env_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        og_env_class = env_class or module.EnvOmniGibson

        env_kwargs = deepcopy(env_meta["env_kwargs"])
        env_kwargs.pop("env_name", None)
        env_kwargs.pop("camera_names", None)
        env_kwargs.pop("camera_height", None)
        env_kwargs.pop("camera_width", None)
        env_kwargs.pop("reward_shaping", None)
        env_kwargs["init_curobo"] = init_curobo
        # MoMaGen encodes robot + dataset difficulty in task names (e.g.
        # r1_picking_up_trash_D0), while OmniGibson's BDDL activity registry
        # only knows the activity name (picking_up_trash). Keep the suffixed
        # env_name for the local wrapper so D0 / D1 / D2 randomization branches
        # still work, but force the underlying OG task lookup to use the BDDL
        # activity name.
        requested_env_name = env_meta["env_name"]
        base_env_name = requested_env_name
        suffix = base_env_name.rsplit("_", 1)[-1]
        if suffix.startswith("D") and suffix[1:].isdigit():
            base_env_name = base_env_name.rsplit("_", 1)[0]
        activity_name = base_env_name
        prefix, sep, remainder = activity_name.partition("_")
        if sep and prefix.startswith("r") and prefix[1:].isdigit():
            activity_name = remainder
        for variant_suffix in ("_mimicgen", "_skillgen"):
            if activity_name.endswith(variant_suffix):
                activity_name = activity_name[: -len(variant_suffix)]
                break
        env_kwargs.setdefault("task", {})["activity_name"] = activity_name
        # Raw / reconstructed BEHAVIOR scene metadata can omit task-level
        # presampled robot poses. In that case current OmniGibson's default
        # BehaviorTask reset path crashes on `robot.model_name in None`; the
        # source HDF5 already contains the desired robot pose in the config, so
        # disable presampled-pose lookup for MoMaGen generation env creation.
        env_kwargs.setdefault("task", {})["use_presampled_robot_pose"] = False
        return og_env_class.create_for_data_processing(
            env_name=env_meta["env_name"],
            policy_rollout=policy_rollout,
            manipulation_only=manipulation_only,
            real_robot_mode=real_robot_mode,
            baseline=baseline,
            **env_kwargs,
        )

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        env_class=env_class,
        camera_names=camera_names, 
        camera_height=camera_height, 
        camera_width=camera_width, 
        reward_shaping=False,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        # init_curobo=init_curobo,
        policy_rollout=policy_rollout,
        manipulation_only=manipulation_only,
        real_robot_mode=real_robot_mode,
        baseline=baseline,
    )

    return env


def make_dataset_video(
    dataset_path,
    video_path,
    num_render=None,
    render_image_names=None,
    use_obs=False,
    video_skip=5,
):
    """
    Helper function to set up args and call @playback_dataset from robomimic
    to get video of generated dataset.
    """
    print("\nmake_dataset_video(\n\tdataset_path={},\n\tvideo_path={},{}\n)".format(
        dataset_path,
        video_path,
        "\n\tnum_render={},".format(num_render) if num_render is not None else "",
    ))
    playback_args = argparse.Namespace()
    playback_args.dataset = dataset_path
    playback_args.filter_key = None
    playback_args.n = num_render
    playback_args.use_obs = use_obs
    playback_args.use_actions = False
    playback_args.render = False
    playback_args.video_path = video_path
    playback_args.video_skip = video_skip
    playback_args.render_image_names = render_image_names
    if (render_image_names is None):
        # default robosuite
        playback_args.render_image_names = ["agentview"]
    playback_args.render_depth_names = None
    playback_args.first = False

    try:
        playback_dataset(playback_args)
    except Exception as e:
        res_str = "playback failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)


def get_default_env_cameras(env_meta):
    """
    Get the default set of cameras for a particular robomimic environment type.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html

    Returns:
        camera_names (list of str): list of camera names that correspond to image observations
    """
    env_type = EnvUtils.get_env_type(env_meta=env_meta)
    if env_type in DEFAULT_CAMERAS:
        return DEFAULT_CAMERAS[env_type]
    # Local MoMaGen registers OmniGibson as robomimic EnvType.OG_TYPE (4), but
    # robomimic's playback DEFAULT_CAMERAS table does not include that key. The
    # OmniGibson wrapper currently ignores camera_name for rgb_array rendering
    # and returns a concatenated ego + viewer frame, so a single stable placeholder
    # is sufficient when a caller asks for default debug-video cameras.
    if env_type == 4:
        return ["agentview"]
    return DEFAULT_CAMERAS[env_type]
