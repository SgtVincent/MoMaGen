from types import SimpleNamespace

import torch as th

from momagen.datagen.waypoint import maybe_apply_phase_routing_target_precontact, select_phase_routing_nav_eef_pose


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
