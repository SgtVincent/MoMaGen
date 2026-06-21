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
    out = {
        "output": path,
        "summary": data.get("summary", {}),
    }
    for key in ("video_output", "obs_video_output", "obs_video_frames"):
        if data.get(key) is not None:
            out[key] = data[key]
    return out


def video_summary(path, frames=None, fps=None, resolution=None, layout=None):
    if not path:
        return None
    out = {
        "path": path,
        "frames": frames,
        "fps": fps,
        "duration_seconds": round(frames / fps, 6) if frames and fps else None,
        "resolution": resolution,
    }
    if layout:
        out["layout"] = layout
    return {k: v for k, v in out.items() if v is not None}


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


def classify_obs_visibility(summary, args):
    visibility = summary.get("observation_visibility") or {}
    result = {
        "passes": [],
        "warnings": [],
        "blockers": [],
        "visibility": visibility,
    }
    if not visibility:
        result["blockers"].append("observation_visibility_metrics_missing")
        return result

    head = visibility.get("head") or {}
    head_marker_rate = head.get("marker_in_frame_rate")
    head_object_fraction = head.get("mean_object_pixel_fraction")
    if isinstance(head_marker_rate, (int, float)):
        if head_marker_rate >= args.min_head_marker_rate:
            result["passes"].append("head_camera_marker_visibility_above_threshold")
        else:
            frames = head.get("frames")
            marker_frames = head.get("marker_in_frame_frames")
            result["warnings"].append(
                f"head_camera_marker_in_frame_only_{marker_frames}_of_{frames}_frames"
            )
            result["blockers"].append("head_camera_contact_marker_visibility_below_threshold")
    else:
        result["blockers"].append("head_camera_marker_visibility_missing")

    if isinstance(head_object_fraction, (int, float)):
        if head_object_fraction >= args.min_head_object_pixel_fraction:
            result["passes"].append("head_camera_radio_pixel_fraction_above_threshold")
        else:
            result["warnings"].append(
                f"head_camera_radio_pixel_fraction_{head_object_fraction:.6f}_below_threshold"
            )
            result["blockers"].append("head_camera_radio_pixel_fraction_below_threshold")
    else:
        result["blockers"].append("head_camera_radio_pixel_fraction_missing")

    wrist_names = [name.strip() for name in args.wrist_cameras.split(",") if name.strip()]
    wrist_ok = []
    for name in wrist_names:
        cam = visibility.get(name) or {}
        marker_rate = cam.get("marker_in_frame_rate")
        if isinstance(marker_rate, (int, float)) and marker_rate >= args.min_wrist_marker_rate:
            wrist_ok.append(name)
    if len(wrist_ok) >= args.min_good_wrist_cameras:
        result["passes"].append("wrist_camera_marker_visibility_above_threshold")
    else:
        result["blockers"].append("insufficient_wrist_camera_marker_visibility")

    return result


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
    obs_layout_long_window = summary_of(args.obs_layout_long_window) if args.obs_layout_long_window else None
    obs_visibility_long_window = summary_of(args.obs_visibility_long_window) if args.obs_visibility_long_window else None

    smoke_summary = smoke["summary"]
    long_summary = long_window["summary"]
    near_summary = (near_checkpoint or {}).get("summary", {})
    obs_layout_summary = (obs_layout_long_window or {}).get("summary", {})
    obs_visibility_summary = (obs_visibility_long_window or {}).get("summary", {})

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

    if obs_layout_long_window is not None:
        if classify_replay(obs_layout_summary) == long_class:
            passes.append("observation_layout_video_reproduces_long_window_metrics")
        else:
            blockers.append("observation_layout_video_replay_disagrees_with_long_window")

    obs_visibility_result = None
    if obs_visibility_long_window is not None:
        obs_visibility_result = classify_obs_visibility(obs_visibility_summary, args)
        passes.extend(obs_visibility_result["passes"])
        warnings.extend(obs_visibility_result["warnings"])
        blockers.extend(obs_visibility_result["blockers"])

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

    if blockers:
        admission = "do_not_admit_yet"
        if obs_visibility_result is not None and any(
            blocker.startswith("head_camera") or blocker.startswith("insufficient_wrist")
            for blocker in obs_visibility_result["blockers"]
        ):
            admission = "keep_as_replay_gated_candidate_pending_observation_quality_review"
    else:
        admission = "admit_candidate_after_human_review"

    return {
        "manifest_type": "momagen_action_replay_admission_gate_v1",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "candidate": args.candidate,
        "dataset": args.dataset,
        "inputs": {
            "smoke": smoke,
            "long_window": long_window,
            "near_checkpoint": near_checkpoint,
            "obs_layout_long_window": obs_layout_long_window,
            "obs_visibility_long_window": obs_visibility_long_window,
        },
        "gate_policy": {
            "short_smoke_state_error_max_abs_threshold": args.smoke_max_abs_threshold,
            "requires_long_window_5_step_hold": True,
            "near_checkpoint_success_is_insufficient_without_long_window_success": True,
            "relaxed_overlap_probe_radius_for_diagnostics": args.relaxed_radius,
            "semantic_video_review_required_before_training_admission": True,
            "observation_visibility_review_required_before_training_admission": args.obs_visibility_long_window is not None,
            "min_head_marker_in_frame_rate": args.min_head_marker_rate,
            "min_head_object_pixel_fraction": args.min_head_object_pixel_fraction,
            "min_wrist_marker_in_frame_rate": args.min_wrist_marker_rate,
            "min_good_wrist_cameras": args.min_good_wrist_cameras,
        },
        "derived": {
            "short_smoke_state_error_max_abs": smoke_err,
            "long_window_class": long_class,
            "near_checkpoint_class": near_class,
            "checkpoint_sensitive": bool(near_class == "pass" and long_class != "pass"),
            "video": video_summary(
                long_window.get("video_output"),
                frames=long_summary.get("num_records"),
                fps=args.video_fps,
                resolution=args.video_resolution,
            ),
            "obs_layout_video": video_summary(
                (obs_layout_long_window or {}).get("obs_video_output"),
                frames=(obs_layout_long_window or {}).get("obs_video_frames") or obs_layout_summary.get("num_records"),
                fps=args.obs_video_fps,
                resolution=args.obs_video_resolution,
                layout="left column: left wrist over right wrist at 224x224 each; right column: head camera at 448x448",
            ),
            "obs_visibility": (obs_visibility_result or {}).get("visibility"),
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
    parser.add_argument("--obs-layout-long-window")
    parser.add_argument("--obs-visibility-long-window")
    parser.add_argument("--long-start", type=int, default=None)
    parser.add_argument("--smoke-max-abs-threshold", type=float, default=0.01)
    parser.add_argument("--relaxed-radius", type=float, default=0.03)
    parser.add_argument("--boundary-hit-sample-limit", type=int, default=6)
    parser.add_argument("--boundary-edge-count", type=int, default=2)
    parser.add_argument("--min-head-marker-rate", type=float, default=0.8)
    parser.add_argument("--min-head-object-pixel-fraction", type=float, default=0.01)
    parser.add_argument("--min-wrist-marker-rate", type=float, default=0.8)
    parser.add_argument("--min-good-wrist-cameras", type=int, default=1)
    parser.add_argument("--wrist-cameras", default="left_wrist,right_wrist")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--obs-video-fps", type=int, default=12)
    parser.add_argument("--video-resolution", default="1280x720")
    parser.add_argument("--obs-video-resolution", default="672x448")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(manifest["quality_verdict"], indent=2))


if __name__ == "__main__":
    main()
