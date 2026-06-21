#!/usr/bin/env python3
"""Build a conservative generated-data admission/conversion preflight manifest.

This check is intentionally no-simulator and no-training. It audits whether a
MoMaGen quality-gated candidate has enough evidence to be queued for the next
BEHAVIOR / openpi-comet admission step, and it fails closed when strict
simulator admission is missing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OPENPI_ALLOWED_GENERATED_ACTION_SOURCES = {
    "demo_ee_trajectory": {"r1pro_23d"},
    "wm_keyposes": {"r1pro_23d"},
    "momagen_bimanual": {"momagen_bimanual_16d", "momagen_16d_to_r1pro_23d"},
    "curobo_plan": {"r1pro_23d"},
    "geometry_base_prior": {"r1pro_23d"},
    "geometry_base_prior_momagen_bimanual": {"r1pro_23d"},
    "osc_diagnostic_fallback": {"r1pro_23d"},
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def inspect_hdf5(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        out["errors"] = ["dataset_missing"]
        return out

    errors: list[str] = []
    with h5py.File(path, "r") as f:
        demo = f.get("data/demo_0")
        if demo is None:
            out["errors"] = ["data_demo_0_missing"]
            return out

        for key in ("actions", "states"):
            ds = demo.get(key)
            if ds is None:
                errors.append(f"{key}_missing")
                continue
            arr = ds[()]
            out[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "finite": bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.number) else None,
            }

        out["datagen_info_keys"] = sorted(demo.get("datagen_info", {}).keys()) if demo.get("datagen_info") else []
        out["sensor_info_keys"] = sorted(demo.get("sensor_info", {}).keys()) if demo.get("sensor_info") else []
        out["data_attrs"] = _summarize_data_attrs(f["data"].attrs) if "data" in f else {}

    actions = out.get("actions") if isinstance(out.get("actions"), dict) else {}
    states = out.get("states") if isinstance(out.get("states"), dict) else {}
    if actions.get("shape", [None, None])[-1] != 23:
        errors.append("actions_not_canonical_r1pro_23d")
    if actions.get("shape", [None])[0] != states.get("shape", [None])[0]:
        errors.append("actions_states_length_mismatch")
    if actions.get("finite") is not True:
        errors.append("actions_not_finite")
    if states.get("finite") is not True:
        errors.append("states_not_finite")
    out["errors"] = errors
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _summarize_data_attrs(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "total" in attrs:
        out["total"] = _json_safe(attrs["total"])
    env_args = attrs.get("env_args")
    if isinstance(env_args, bytes):
        env_args = env_args.decode("utf-8", errors="replace")
    if isinstance(env_args, str):
        try:
            parsed = json.loads(env_args)
        except json.JSONDecodeError:
            out["env_args_parse_error"] = True
        else:
            env_kwargs = parsed.get("env_kwargs") if isinstance(parsed.get("env_kwargs"), dict) else {}
            task = env_kwargs.get("task") if isinstance(env_kwargs.get("task"), dict) else {}
            robots = env_kwargs.get("robots") if isinstance(env_kwargs.get("robots"), list) else []
            scene = env_kwargs.get("scene") if isinstance(env_kwargs.get("scene"), dict) else {}
            out["env_args_summary"] = {
                "env_name": parsed.get("env_name"),
                "env_version": parsed.get("env_version"),
                "task_activity_name": task.get("activity_name"),
                "task_activity_definition_id": task.get("activity_definition_id"),
                "task_activity_instance_id": task.get("activity_instance_id"),
                "robot_types": [robot.get("type") for robot in robots if isinstance(robot, dict)],
                "scene_model": scene.get("scene_model"),
                "scene_instance": scene.get("scene_instance"),
                "init_curobo": env_kwargs.get("init_curobo"),
            }
    return out


def check_file(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    out = {
        "path": str(p),
        "exists": p.is_file(),
        "bytes": p.stat().st_size if p.is_file() else None,
    }
    return out


def strict_admission_checks(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "available": False,
            "passes": [],
            "failures": ["strict_simulator_admission_report_missing"],
        }

    admission = report.get("admission") if isinstance(report.get("admission"), dict) else {}
    verifier = report.get("verifier") if isinstance(report.get("verifier"), dict) else {}
    rollout = report.get("rollout_metadata") if isinstance(report.get("rollout_metadata"), dict) else {}
    checks = {
        "admission.accepted": admission.get("accepted") is True,
        "admission.stage": admission.get("stage") == "p0_simulator_verifier_admission",
        "admission.strict_generated_data_gate": admission.get("strict_generated_data_gate") is True,
        "admission.requires_simulator_verifier": admission.get("requires_simulator_verifier") is False,
        "verifier.accepted": verifier.get("accepted") is True,
        "verifier.strict_generated_data_gate": verifier.get("strict_generated_data_gate") is True,
        "verifier.rollout_attempted": verifier.get("rollout_attempted") is True or rollout.get("rollout_attempted") is True,
        "verifier.physics_ok": verifier.get("physics_ok") is True,
        "verifier.contact_ok": verifier.get("contact_ok") is True,
        "verifier.predicate_ok": verifier.get("predicate_ok") is True,
        "verifier.predicate_trace_steps": int(verifier.get("predicate_trace_steps") or 0) > 0,
    }
    return {
        "available": True,
        "passes": [key for key, passed in checks.items() if passed],
        "failures": [key for key, passed in checks.items() if not passed],
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset)
    gate_path = Path(args.momagen_gate)
    gate = load_json(gate_path)
    quality = gate.get("quality_verdict") if isinstance(gate.get("quality_verdict"), dict) else {}
    derived = gate.get("derived") if isinstance(gate.get("derived"), dict) else {}
    human_review = quality.get("human_semantic_review") if isinstance(quality.get("human_semantic_review"), dict) else {}

    hdf5 = inspect_hdf5(dataset)
    strict_report = load_json(Path(args.strict_admission_report)) if args.strict_admission_report else None
    strict = strict_admission_checks(strict_report)

    action_source = "r1pro_23d" if nested(hdf5, "actions", "shape", default=[])[-1:] == [23] else "unknown"
    proposed_source_type = args.proposed_source_type
    allowed_actions = OPENPI_ALLOWED_GENERATED_ACTION_SOURCES.get(proposed_source_type)
    lineage_compatible = allowed_actions is not None and action_source in allowed_actions

    videos = {
        "third_view": check_file(nested(derived, "video", "path")),
        "obs_layout": check_file(nested(derived, "obs_layout_video", "path")),
    }
    video_failures = [
        f"{name}_video_missing_or_empty"
        for name, info in videos.items()
        if not info or info.get("exists") is not True or not info.get("bytes")
    ]

    blockers: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    if hdf5.get("errors"):
        blockers.extend(f"hdf5_{err}" for err in hdf5["errors"])
    else:
        passes.append("hdf5_actions_states_canonical_and_finite")

    if quality.get("admission_recommendation") == "admit_observation_qualified_seed_for_generated_data_pipeline":
        passes.append("momagen_action_observation_gate_admits_seed")
    else:
        blockers.append("momagen_action_observation_gate_not_admitted")

    if human_review.get("status") == "passed":
        passes.append("human_semantic_review_passed")
    else:
        blockers.append("human_semantic_review_missing_or_not_passed")

    if video_failures:
        blockers.extend(video_failures)
    else:
        passes.append("review_videos_present")

    critical = derived.get("obs_visibility_critical_window")
    if isinstance(critical, dict) and critical.get("visibility"):
        passes.append("critical_window_visibility_metrics_present")
    else:
        blockers.append("critical_window_visibility_metrics_missing")

    if strict["failures"]:
        blockers.extend(strict["failures"])
    else:
        passes.append("strict_behavior_simulator_admission_passed")

    if not lineage_compatible:
        blockers.append("openpi_generated_data_lineage_mapping_unresolved")

    warnings.extend(quality.get("warnings") if isinstance(quality.get("warnings"), list) else [])

    conversion_eligible = not blockers
    if conversion_eligible:
        status = "strict_admission_ready_for_training_candidate_manifest"
    elif (
        "momagen_action_observation_gate_admits_seed" in passes
        and "human_semantic_review_passed" in passes
        and "critical_window_visibility_metrics_present" in passes
    ):
        status = "observation_qualified_not_conversion_eligible"
    else:
        status = "blocked_before_observation_qualified_seed"

    return {
        "manifest_type": "momagen_generated_data_admission_preflight_v1",
        "generated_at_utc": utc_now_iso(),
        "candidate": args.candidate,
        "task": args.task,
        "status": status,
        "conversion_eligible": conversion_eligible,
        "auto_training_enabled": False,
        "real_training_enabled": False,
        "training_started": False,
        "checkpoint_written": False,
        "wandb_enabled": False,
        "inputs": {
            "dataset": str(dataset),
            "momagen_gate": str(gate_path),
            "strict_admission_report": args.strict_admission_report,
        },
        "hdf5": hdf5,
        "momagen_quality_gate": {
            "manifest_type": gate.get("manifest_type"),
            "candidate": gate.get("candidate"),
            "admission_recommendation": quality.get("admission_recommendation"),
            "passes": quality.get("passes", []),
            "warnings": quality.get("warnings", []),
            "blockers": quality.get("blockers", []),
            "human_semantic_review": human_review,
            "critical_window": derived.get("obs_visibility_critical_window"),
        },
        "media": videos,
        "strict_behavior_admission": strict,
        "openpi_lineage_preflight": {
            "proposed_source_type": proposed_source_type,
            "action_source": action_source,
            "lineage_compatible_with_current_openpi_contract": lineage_compatible,
            "allowed_action_sources_for_proposed_type": sorted(allowed_actions) if allowed_actions else None,
            "note": (
                "A MoMaGen R1Pro-23D candidate still needs an explicit source_type/action_source mapping "
                "accepted by openpi-comet before training-candidate manifest generation."
            ),
        },
        "verdict": {
            "passes": sorted(set(passes)),
            "warnings": sorted(set(warnings)),
            "blockers": sorted(set(blockers)),
        },
        "next_actions": [
            "Materialize a BEHAVIOR strict simulator admission report for the A201 replay/action artifact.",
            "Resolve or add the openpi-comet generated_data_lineage source_type/action_source mapping for MoMaGen R1Pro-23D output.",
            "Only after strict admission and lineage compatibility pass, emit a b1k_generated_data_training_candidate manifest for loader/conversion smoke.",
            "Reuse this preflight on A217/A218 or historical successful candidates before scheduling expensive replay/admission work.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--momagen-gate", required=True)
    parser.add_argument("--strict-admission-report")
    parser.add_argument(
        "--proposed-source-type",
        default="momagen_r1pro_23d",
        help="Source type to test against the current openpi-comet generated-data lineage contract.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(manifest["verdict"], indent=2))


if __name__ == "__main__":
    main()
