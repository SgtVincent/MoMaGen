#!/usr/bin/env python3
"""Build a conservative MoMaGen action-replay admission manifest.

The gate intentionally separates three evidence levels:
1. short smoke replay state consistency,
2. longer pre-contact / press-window replay,
3. near-contact checkpoint replay.

If the near-contact checkpoint succeeds but the longer window fails, the
candidate is still blocked because the generated trajectory is sensitive to
small accumulated simulation differences.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summary_of(path):
    data = load_json(path)
    return {
        "output": path,
        "summary": data.get("summary", {}),
    }


def state_error_max(summary):
    err = summary.get("state_error") or {}
    return err.get("max_abs_max")


def replay_has_toggle(summary):
    return summary.get("first_toggle_value_step") is not None and (summary.get("max_robot_can_toggle_steps") or 0) >= 5


def classify_replay(summary):
    if replay_has_toggle(summary):
        return "pass"
    if (summary.get("max_robot_can_toggle_steps") or 0) > 0:
        return "partial"
    return "fail"


def find_radius_boundary_steps(path, primary_radius=None, relaxed_radius=0.03, sample_hit_limit=6):
    data = load_json(path)
    out = []
    for record in data.get("records", []):
        for obj_name, obj in (record.get("objects") or {}).items():
            marker_radius = obj.get("marker_radius")
            target_primary_radius = primary_radius if primary_radius is not None else marker_radius
            primary = None
            relaxed = None
            for probe in obj.get("overlap_probes") or []:
                radius = probe.get("radius")
                if target_primary_radius is not None and abs(radius - target_primary_radius) < 1e-9:
                    primary = probe
                if abs(radius - relaxed_radius) < 1e-9:
                    relaxed = probe
            if (
                primary
                and relaxed
                and not primary.get("valid_robot_finger_hit")
                and relaxed.get("valid_robot_finger_hit")
            ):
                out.append(
                    {
                        "step": record.get("step"),
                        "object": obj_name,
                        "marker_radius": marker_radius,
                        "relaxed_radius": relaxed_radius,
                        "left_finger_dist": ((obj.get("finger_min_dist_to_marker") or {}).get("left") or {}).get("dist"),
                        "primary_num_hits": primary.get("num_hits"),
                        "relaxed_num_hits": relaxed.get("num_hits"),
                        "primary_hits_sample": primary.get("hits", [])[:sample_hit_limit],
                        "relaxed_hits_sample": relaxed.get("hits", [])[:sample_hit_limit],
                    }
                )
    return out


def compact_boundary_steps(boundary_steps, focus_steps=None, edge_count=3):
    if not boundary_steps:
        return {
            "count": 0,
            "first_step": None,
            "last_step": None,
            "steps": [],
            "examples": [],
        }
    focus_steps = set(focus_steps or [])
    by_step = {entry["step"]: entry for entry in boundary_steps}
    examples = []
    used = set()
    for step in sorted(focus_steps):
        if step in by_step:
            examples.append(by_step[step])
            used.add(step)
    edge_entries = boundary_steps[:edge_count] + boundary_steps[-edge_count:]
    for entry in edge_entries:
        if entry["step"] not in used:
            examples.append(entry)
            used.add(entry["step"])
    return {
        "count": len(boundary_steps),
        "first_step": boundary_steps[0]["step"],
        "last_step": boundary_steps[-1]["step"],
        "steps": [entry["step"] for entry in boundary_steps],
        "examples": examples,
    }


def build_manifest(args):
    smoke = summary_of(args.smoke)
    long_window = summary_of(args.long_window)
    near_checkpoint = summary_of(args.near_checkpoint) if args.near_checkpoint else None

    smoke_summary = smoke["summary"]
    long_summary = long_window["summary"]
    near_summary = (near_checkpoint or {}).get("summary", {})

    passes = []
    warnings = []
    blockers = []

    smoke_err = state_error_max(smoke_summary)
    if isinstance(smoke_err, (int, float)) and smoke_err <= args.smoke_max_abs_threshold:
        passes.append("short_smoke_state_consistent")
    else:
        blockers.append("short_smoke_state_inconsistent")

    long_class = classify_replay(long_summary)
    if long_class == "pass":
        passes.append("long_precontact_window_reproduces_5_step_hold")
    elif long_class == "partial":
        blockers.append("long_precontact_window_only_partial_hold")
    else:
        blockers.append("long_precontact_window_no_predicate_hold")

    near_class = None
    if near_checkpoint is not None:
        near_class = classify_replay(near_summary)
        if near_class == "pass":
            passes.append("near_contact_checkpoint_reproduces_5_step_hold")
        elif near_class == "partial":
            warnings.append("near_contact_checkpoint_only_partial_hold")
        else:
            warnings.append("near_contact_checkpoint_no_predicate_hold")

    if near_class == "pass" and long_class != "pass":
        blockers.append("checkpoint_sensitive_replay_disagreement")

    boundary_steps = find_radius_boundary_steps(
        args.long_window,
        relaxed_radius=args.relaxed_radius,
        sample_hit_limit=args.boundary_hit_sample_limit,
    )
    boundary_summary = compact_boundary_steps(
        boundary_steps,
        focus_steps=[long_summary.get("first_can_toggle_step"), long_summary.get("best_left_finger_dist", {}).get("step"), args.long_start],
        edge_count=args.boundary_edge_count,
    )
    if boundary_steps:
        warnings.append("primary_overlap_radius_boundary_sensitive")
    else:
        passes.append("no_primary_to_relaxed_overlap_boundary_steps_observed")

    if long_summary.get("first_task_success_step") == args.long_start:
        warnings.append("initial_snapshot_task_success_likely_predicate_cache_artifact")

    admission = "do_not_admit_yet" if blockers else "admit_candidate_after_human_review"

    return {
        "manifest_type": "momagen_action_replay_admission_gate_v1",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "candidate": args.candidate,
        "dataset": args.dataset,
        "inputs": {
            "smoke": smoke,
            "long_window": long_window,
            "near_checkpoint": near_checkpoint,
        },
        "gate_policy": {
            "short_smoke_state_error_max_abs_threshold": args.smoke_max_abs_threshold,
            "requires_long_window_5_step_hold": True,
            "near_checkpoint_success_is_insufficient_without_long_window_success": True,
            "relaxed_overlap_probe_radius_for_diagnostics": args.relaxed_radius,
        },
        "derived": {
            "short_smoke_state_error_max_abs": smoke_err,
            "long_window_class": long_class,
            "near_checkpoint_class": near_class,
            "checkpoint_sensitive": bool(near_class == "pass" and long_class != "pass"),
            "primary_to_relaxed_overlap_boundary_steps": boundary_summary,
        },
        "quality_verdict": {
            "passes": sorted(set(passes)),
            "warnings": sorted(set(warnings)),
            "blockers": sorted(set(blockers)),
            "admission_recommendation": admission,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--long-window", required=True)
    parser.add_argument("--near-checkpoint")
    parser.add_argument("--long-start", type=int, default=None)
    parser.add_argument("--smoke-max-abs-threshold", type=float, default=0.01)
    parser.add_argument("--relaxed-radius", type=float, default=0.03)
    parser.add_argument("--boundary-hit-sample-limit", type=int, default=6)
    parser.add_argument("--boundary-edge-count", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(manifest["quality_verdict"], indent=2))


if __name__ == "__main__":
    main()
