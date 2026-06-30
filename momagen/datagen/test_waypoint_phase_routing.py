from types import SimpleNamespace

import torch as th

from momagen.datagen.waypoint import maybe_apply_phase_routing_target_precontact


class _FakeObject:
    states = {}

    def __init__(self, pos):
        self._pos = th.tensor(pos, dtype=th.float32)

    def get_position_orientation(self):
        return self._pos, th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)


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
