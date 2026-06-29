#!/usr/bin/env python3
"""Build a fail-closed generated trajectory quality gate.

This gate consumes the no-simulator metrics exported by
``momagen/scripts/debug/export_generated_trajectory_quality.py``. It is meant to
catch complete but visually unreviewable contact-rich episodes where endpoint
or task success hides large object / end-effector loops.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_CRITICAL_RANGES = ("phase2", "phase2_after_mp", "phase3", "phase3_after_mp")
DEFAULT_REQUIRE_TRACKS = ("radio_89", "left_eef", "right_eef", "base_xy", "base_yaw")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_range(metrics: dict[str, Any], label: str) -> dict[str, Any] | None:
    for item in metrics.get("ranges", []):
        if isinstance(item, dict) and item.get("label") == label:
            return item
    return None


def track_metric(range_entry: dict[str, Any], track: str, key: str) -> float | None:
    tracks = range_entry.get("tracks") if isinstance(range_entry.get("tracks"), dict) else {}
    values = tracks.get(track) if isinstance(tracks.get(track), dict) else {}
    value = values.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def add_threshold_check(
    *,
    passes: list[str],
    blockers: list[str],
    warnings: list[str],
    details: list[dict[str, Any]],
    range_entry: dict[str, Any],
    track: str,
    metric_key: str,
    threshold: float,
    blocker_name: str,
    pass_name: str,
    warning_only: bool = False,
) -> None:
    label = str(range_entry.get("label"))
    value = track_metric(range_entry, track, metric_key)
    detail = {
        "range": label,
        "track": track,
        "metric": metric_key,
        "value": value,
        "threshold": threshold,
        "warning_only": warning_only,
    }
    details.append(detail)
    if value is None:
        key = f"{label}_{track}_{metric_key}_missing"
        (warnings if warning_only else blockers).append(key)
        detail["status"] = "missing"
        return
    if value <= threshold:
        passes.append(pass_name)
        detail["status"] = "pass"
    else:
        (warnings if warning_only else blockers).append(blocker_name)
        detail["status"] = "fail"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = Path(args.metrics)
    metrics = load_json(metrics_path)
    critical_ranges = split_csv(args.critical_ranges)
    require_tracks = split_csv(args.require_tracks)

    passes: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    details: list[dict[str, Any]] = []
    range_summaries: dict[str, Any] = {}

    for label in critical_ranges:
        range_entry = find_range(metrics, label)
        if range_entry is None:
            blockers.append(f"{label}_range_missing")
            range_summaries[label] = {"available": False}
            continue

        tracks = range_entry.get("tracks") if isinstance(range_entry.get("tracks"), dict) else {}
        missing_tracks = [track for track in require_tracks if track not in tracks]
        if missing_tracks:
            blockers.append(f"{label}_required_tracks_missing")
        else:
            passes.append(f"{label}_required_tracks_present")
        range_summaries[label] = {
            "available": True,
            "start": range_entry.get("start"),
            "end": range_entry.get("end"),
            "missing_tracks": missing_tracks,
        }

    phase2 = find_range(metrics, "phase2")
    phase2_after = find_range(metrics, "phase2_after_mp")
    phase3_after = find_range(metrics, "phase3_after_mp")

    if phase2:
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2,
            track=args.held_object_track,
            metric_key="path_to_net_ratio",
            threshold=args.max_phase_object_path_net,
            blocker_name="phase2_held_object_path_net_too_high",
            pass_name="phase2_held_object_path_net_within_limit",
        )
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2,
            track="left_eef",
            metric_key="path_to_net_ratio",
            threshold=args.max_phase_eef_path_net,
            blocker_name="phase2_left_eef_path_net_too_high",
            pass_name="phase2_left_eef_path_net_within_limit",
        )
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2,
            track="right_eef",
            metric_key="path_to_net_ratio",
            threshold=args.max_phase_eef_path_net,
            blocker_name="phase2_right_eef_path_net_too_high",
            pass_name="phase2_right_eef_path_net_within_limit",
        )

    if phase2_after:
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2_after,
            track=args.held_object_track,
            metric_key="path_m",
            threshold=args.max_after_mp_object_path_m,
            blocker_name="phase2_after_mp_held_object_path_too_long",
            pass_name="phase2_after_mp_held_object_path_within_limit",
        )
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2_after,
            track=args.held_object_track,
            metric_key="path_to_net_ratio",
            threshold=args.max_after_mp_object_path_net,
            blocker_name="phase2_after_mp_held_object_path_net_too_high",
            pass_name="phase2_after_mp_held_object_path_net_within_limit",
        )
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2_after,
            track="right_eef",
            metric_key="path_to_net_ratio",
            threshold=args.max_after_mp_eef_path_net,
            blocker_name="phase2_after_mp_right_eef_path_net_too_high",
            pass_name="phase2_after_mp_right_eef_path_net_within_limit",
        )
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase2_after,
            track="base_xy",
            metric_key="path_m",
            threshold=args.max_after_mp_base_path_m,
            blocker_name="phase2_after_mp_base_path_too_long",
            pass_name="phase2_after_mp_base_path_within_limit",
        )

    if phase3_after:
        add_threshold_check(
            passes=passes,
            blockers=blockers,
            warnings=warnings,
            details=details,
            range_entry=phase3_after,
            track=args.held_object_track,
            metric_key="path_to_net_ratio",
            threshold=args.max_phase3_after_mp_object_path_net,
            blocker_name="phase3_after_mp_held_object_path_net_too_high",
            pass_name="phase3_after_mp_held_object_path_net_within_limit",
        )

    if args.camera_framing_reviewed:
        passes.append("camera_framing_human_review_recorded")
    elif args.require_camera_framing_review:
        blockers.append("camera_framing_human_review_missing")
    else:
        warnings.append("camera_framing_human_review_missing")

    accepted = not blockers
    if accepted:
        recommendation = "trajectory_quality_passed_pending_semantic_review"
    else:
        recommendation = "do_not_admit_trajectory_quality_failed"

    return {
        "manifest_type": "momagen_generated_trajectory_quality_gate_v1",
        "created_at_utc": utc_now_iso(),
        "candidate": args.candidate,
        "task": args.task,
        "accepted": accepted,
        "admission_recommendation": recommendation,
        "inputs": {
            "metrics": str(metrics_path),
            "dataset": metrics.get("dataset"),
            "tracked_objects": metrics.get("tracked_objects", []),
        },
        "gate_policy": {
            "critical_ranges": critical_ranges,
            "required_tracks": require_tracks,
            "held_object_track": args.held_object_track,
            "max_phase_object_path_net": args.max_phase_object_path_net,
            "max_phase_eef_path_net": args.max_phase_eef_path_net,
            "max_after_mp_object_path_m": args.max_after_mp_object_path_m,
            "max_after_mp_object_path_net": args.max_after_mp_object_path_net,
            "max_after_mp_eef_path_net": args.max_after_mp_eef_path_net,
            "max_after_mp_base_path_m": args.max_after_mp_base_path_m,
            "max_phase3_after_mp_object_path_net": args.max_phase3_after_mp_object_path_net,
            "require_camera_framing_review": args.require_camera_framing_review,
            "camera_framing_reviewed": args.camera_framing_reviewed,
        },
        "range_summaries": range_summaries,
        "checks": details,
        "quality_verdict": {
            "passes": sorted(set(passes)),
            "warnings": sorted(set(warnings)),
            "blockers": sorted(set(blockers)),
        },
        "next_actions": [
            "Reject this candidate for admission if blockers are present, even when task_success or success_rate passed.",
            "Review trajectory_topdown.png and trajectory_timeseries.png for the first failing range.",
            "Localize whether the failing after-MP segment comes from replay, contact prealign, post-MP press, or held-object preservation.",
            "Only rerun semantic video review after this gate passes with write_video=True outputs under momagen/datasets/generated/.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--metrics", required=True, help="trajectory_quality_metrics.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-object-track", default="radio_89")
    parser.add_argument("--critical-ranges", default=",".join(DEFAULT_CRITICAL_RANGES))
    parser.add_argument("--require-tracks", default=",".join(DEFAULT_REQUIRE_TRACKS))
    parser.add_argument("--max-phase-object-path-net", type=float, default=5.0)
    parser.add_argument("--max-phase-eef-path-net", type=float, default=6.0)
    parser.add_argument("--max-after-mp-object-path-m", type=float, default=1.25)
    parser.add_argument("--max-after-mp-object-path-net", type=float, default=6.0)
    parser.add_argument("--max-after-mp-eef-path-net", type=float, default=8.0)
    parser.add_argument("--max-after-mp-base-path-m", type=float, default=1.25)
    parser.add_argument("--max-phase3-after-mp-object-path-net", type=float, default=10.0)
    parser.add_argument(
        "--require-camera-framing-review",
        action="store_true",
        help="Fail closed until a human/vision check confirms the robot stays reviewable in camera.",
    )
    parser.add_argument(
        "--camera-framing-reviewed",
        action="store_true",
        help="Record that camera-framing review passed outside this no-simulator gate.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(manifest["quality_verdict"], indent=2))


if __name__ == "__main__":
    main()
