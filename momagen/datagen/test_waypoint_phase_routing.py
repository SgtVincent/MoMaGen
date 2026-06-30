from types import SimpleNamespace

import torch as th
import omnigibson.utils.transform_utils as T

from momagen.datagen.waypoint import (
    maybe_apply_phase_routing_target_precontact,
    maybe_apply_phase_routing_target_precontact_to_maps,
    select_phase_routing_nav_eef_pose,
)


class _FakeObject:
    states = {}

    def __init__(self, pos):
        self._pos = th.tensor(pos, dtype=th.float32)

    def get_position_orientation(self):
        return self._pos, th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)


class _FakeLink:
    def __init__(self, name, body_name):
        self.name = name
        self.body_name = body_name

    def get_position_orientation(self):
        return (
            th.tensor([0.5, 0.0, 0.0], dtype=th.float32),
            th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
        )


def test_phase_routing_target_precontact_disabled(monkeypatch):
    monkeypatch.delenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", raising=False)
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert adjusted is target_pose
    assert record is None


def test_phase_routing_target_precontact_moves_active_arm_away_from_ref(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_DISTANCE", "0.2")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_Z", "0.05")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    phase_logs = {}
    target_pose = {
        "left": (th.tensor([0.0, 1.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
        "right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
    }

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=3),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
        object_ref={"arm_right": "radio_89"},
        phase_type="coordinated",
        phase_logs=phase_logs,
    )

    assert adjusted["left"][0] is target_pose["left"][0]
    assert th.allclose(adjusted["right"][0], th.tensor([1.2, 0.0, 0.05]))
    assert record["applied"] is True
    assert record["arms"][0]["reason"] == "arm_not_active"
    assert phase_logs[3]["phase_routing_target_precontact"] == [record]


def test_phase_routing_target_precontact_can_use_explicit_approach_vector(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_DISTANCE", "0.2")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "0,1,0")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert th.allclose(adjusted["right"][0], th.tensor([1.0, 0.2, 0.0]))
    assert record["approach_vector"] == [0.0, 1.0, 0.0]
    assert record["approach_vector_frame"] == "world"
    assert record["arms"][0]["approach_vector_world"] == [0.0, 1.0, 0.0]


def test_phase_routing_target_precontact_approach_vector_rejects_zero(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "0,0,0")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert adjusted["right"][0] is target_pose["right"][0]
    assert record["applied"] is False
    assert record["reason"] == "no_arm_adjusted"
    assert record["arms"][0]["reason"] == "zero_approach_vector"


def test_phase_routing_target_precontact_respects_phase_gate(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_MIN_PHASE", "2")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=1),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert adjusted is target_pose
    assert record["applied"] is False
    assert record["reason"] == "phase_out_of_range"


def test_phase_routing_target_precontact_can_add_finger_link_goal(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_DISTANCE", "0.1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_DISTANCE", "0.03")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_Z", "0.02")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_MARKER_LOCAL_OFFSET", "0.01,0.02,0.0")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert th.allclose(adjusted["right"][0], th.tensor([1.1, 0.0, 0.0]))
    assert "right_gripper_finger_link1" in adjusted
    assert th.allclose(adjusted["right_gripper_finger_link1"][0], th.tensor([0.04, 0.02, 0.02]))
    assert th.allclose(adjusted["right_gripper_finger_link1"][1], quat)
    assert record["arms"][0]["finger_link_goal"]["applied"] is True
    assert record["arms"][0]["finger_link_goal"]["link"] == "right_gripper_finger_link1"
    assert record["arms"][0]["finger_link_goal"]["distance"] == 0.03
    assert record["arms"][0]["finger_link_goal"]["marker_local_offset"] == [0.01, 0.02, 0.0]


def test_phase_routing_target_precontact_nav_can_override_finger_link_distance(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_DISTANCE", "0.03")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_FINGER_LINK_GOAL_DISTANCE", "0.12")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
        phase_type="navigation",
    )

    assert th.allclose(adjusted["right_gripper_finger_link1"][0], th.tensor([0.12, 0.0, 0.0]))
    assert record["finger_link_goal_distance"] == 0.12
    assert record["arms"][0]["finger_link_goal"]["distance"] == 0.12


def test_phase_routing_target_precontact_nav_distance_override_is_navigation_only(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL_DISTANCE", "0.03")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_FINGER_LINK_GOAL_DISTANCE", "0.12")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
        phase_type="uncoordinated",
    )

    assert th.allclose(adjusted["right_gripper_finger_link1"][0], th.tensor([0.03, 0.0, 0.0]))
    assert record["finger_link_goal_distance"] == 0.03


def test_phase_routing_target_precontact_can_add_clearance_link_goal(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_CLEARANCE_LINKS", "right_arm_link4:0.15,-0.05,0.25")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    link = _FakeLink("robot:right_arm_link4", "right_arm_link4")
    robot = SimpleNamespace(links={"right_arm_link4": link})

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.5, 0.5, 0.1]),
        phase_type="navigation",
    )

    assert "right_arm_link4" in adjusted
    assert th.allclose(adjusted["right_arm_link4"][0], th.tensor([0.65, 0.45, 0.35]))
    assert th.allclose(adjusted["right_arm_link4"][1], quat)
    assert record["clearance_link_frame"] == "world"
    assert record["arms"][0]["clearance_link_goals"][0]["applied"] is True
    assert record["arms"][0]["clearance_link_goals"][0]["link"] == "right_arm_link4"


def test_phase_routing_target_precontact_can_add_posture_constraint(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "1,0,0")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE_AXIS", "-z")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    target_rot = th.as_tensor(T.quat2mat(adjusted["right"][1]))
    assert th.allclose(-target_rot[:, 2], th.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert record["posture"]["enabled"] is True
    assert record["posture"]["records"][0]["applied"] is True
    assert record["posture"]["records"][0]["link"] == "right"
    assert record["posture"]["records"][0]["axis"] == "-z"


def test_phase_routing_target_precontact_posture_can_target_explicit_link(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "0,1,0")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE_LINKS", "right_gripper_finger_link1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE_AXIS", "x")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    target_rot = th.as_tensor(T.quat2mat(adjusted["right_gripper_finger_link1"][1]))
    assert th.allclose(target_rot[:, 0], th.tensor([0.0, 1.0, 0.0]), atol=1e-5)
    assert record["posture"]["records"][0]["link"] == "right_gripper_finger_link1"
    assert record["posture"]["records"][0]["applied"] is True


def test_phase_routing_target_precontact_posture_without_approach_fails_closed(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE", "1")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
    )

    assert adjusted["right"][1] is target_pose["right"][1]
    assert record["posture"]["records"][0] == {
        "link": "right",
        "applied": False,
        "reason": "missing_approach_vector",
    }


def test_phase_routing_target_precontact_map_adapter_adds_new_link_targets(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "0,1,0")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE_LINKS", "right_gripper_finger_link1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_POSTURE_AXIS", "x")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pos = {"right": th.tensor([1.0, 0.0, 0.0])}
    target_quat = {"right": quat}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    record = maybe_apply_phase_routing_target_precontact_to_maps(
        target_pos,
        target_quat,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
        phase_type="coordinated",
    )

    assert record["applied"] is True
    assert "right_gripper_finger_link1" in target_pos
    assert "right_gripper_finger_link1" in target_quat
    target_rot = th.as_tensor(T.quat2mat(target_quat["right_gripper_finger_link1"]))
    assert th.allclose(target_rot[:, 0], th.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_phase_routing_target_precontact_map_adapter_maps_eef_link_to_arm(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_APPROACH_VECTOR", "0,1,0")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FINGER_LINK_GOAL", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_FORCE_FINGER_LINK", "right_gripper_finger_link1")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pos = {"right_eef_link": th.tensor([1.0, 0.0, 0.0])}
    target_quat = {"right_eef_link": quat}
    finger_link = _FakeLink(
        "robot:right_gripper_finger_link1",
        "right_gripper_finger_link1",
    )
    robot = SimpleNamespace(
        eef_link_names={"right": "right_eef_link"},
        finger_links={"right": [finger_link]},
        links={"right_gripper_finger_link1": finger_link},
    )

    record = maybe_apply_phase_routing_target_precontact_to_maps(
        target_pos,
        target_quat,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.0, 0.0, 0.0]),
        phase_type="coordinated",
    )

    assert record["applied"] is True
    assert record["arms"][0]["arm"] == "right_eef_link"
    assert record["arms"][0]["applied"] is True
    assert "right_gripper_finger_link1" in target_pos


def test_phase_routing_clearance_link_goal_flows_to_explicit_nav_policy(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_CLEARANCE_LINKS", "right_arm_link4:0.15,-0.05,0.25")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_TARGET_POLICY", "explicit_links_only")
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), quat)}
    link = _FakeLink("robot:right_arm_link4", "right_arm_link4")
    robot = SimpleNamespace(links={"right_arm_link4": link})

    adjusted, _ = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.5, 0.5, 0.1]),
        phase_type="navigation",
    )

    selected = select_phase_routing_nav_eef_pose(adjusted, "right")

    assert selected == {"right_arm_link4": adjusted["right_arm_link4"]}


def test_phase_routing_target_precontact_clearance_link_goal_requires_robot_link(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT_ARMS", "right")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_CLEARANCE_LINKS", "right_arm_link4:0.15,-0.05,0.25")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}
    robot = SimpleNamespace(links={})

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0, robot=robot),
        ref_obj=_FakeObject([0.5, 0.5, 0.1]),
        phase_type="navigation",
    )

    assert "right_arm_link4" not in adjusted
    assert record["arms"][0]["clearance_link_goals"][0] == {
        "link": "right_arm_link4",
        "applied": False,
        "reason": "missing_robot_link",
    }


def test_phase_routing_target_precontact_clearance_link_specs_fail_closed(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_PRECONTACT", "1")
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_TARGET_CLEARANCE_LINKS", "right_arm_link4:0.1")
    target_pose = {"right": (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    adjusted, record = maybe_apply_phase_routing_target_precontact(
        target_pose,
        env=SimpleNamespace(execution_phase_ind=0),
        ref_obj=_FakeObject([0.5, 0.5, 0.1]),
    )

    assert adjusted is target_pose
    assert record["applied"] is False
    assert "MOMAGEN_PHASE_ROUTING_TARGET_CLEARANCE_LINKS" in record["reason"]


def test_select_phase_routing_nav_eef_pose_preserves_explicit_link_targets():
    left_pose = (th.tensor([0.0, 1.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    right_pose = (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    finger_pose = (th.tensor([0.2, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    eef_pose = {
        "left": left_pose,
        "right": right_pose,
        "right_gripper_finger_link1": finger_pose,
    }

    selected = select_phase_routing_nav_eef_pose(eef_pose, "right")

    assert selected == {"right": right_pose, "right_gripper_finger_link1": finger_pose}


def test_select_phase_routing_nav_eef_pose_can_use_explicit_links_only(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_TARGET_POLICY", "explicit_links_only")
    left_pose = (th.tensor([0.0, 1.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    right_pose = (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    finger_pose = (th.tensor([0.2, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    eef_pose = {
        "left": left_pose,
        "right": right_pose,
        "right_gripper_finger_link1": finger_pose,
    }

    selected = select_phase_routing_nav_eef_pose(eef_pose, "right")

    assert selected == {"right_gripper_finger_link1": finger_pose}


def test_select_phase_routing_nav_eef_pose_explicit_links_only_falls_back_to_arm(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_TARGET_POLICY", "explicit_links_only")
    right_pose = (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    selected = select_phase_routing_nav_eef_pose({"right": right_pose}, "right")

    assert selected == {"right": right_pose}


def test_select_phase_routing_nav_eef_pose_rejects_unknown_policy(monkeypatch):
    monkeypatch.setenv("MOMAGEN_PHASE_ROUTING_NAV_TARGET_POLICY", "unknown")
    right_pose = (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    try:
        select_phase_routing_nav_eef_pose({"right": right_pose}, "right")
    except ValueError as exc:
        assert "MOMAGEN_PHASE_ROUTING_NAV_TARGET_POLICY" in str(exc)
    else:
        raise AssertionError("unknown nav target policies must fail closed")
