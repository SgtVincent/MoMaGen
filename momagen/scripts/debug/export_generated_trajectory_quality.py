#!/usr/bin/env python3
"""Export trajectory-quality diagnostics for a generated MoMaGen HDF5.

This is an offline diagnostic. It does not restore OmniGibson state or require
Isaac Sim. It reads recorded datagen_info poses and summarizes path efficiency
for base, left/right EEs, and tracked objects. The output is intended for
debugging contact-rich bimanual runs where endpoint success can hide poor
trajectory quality.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OBJECTS = ("radio_89", "coffee_table_koagbh_0")


def _load_demo(path, demo_key="demo_0"):
    with h5py.File(path, "r") as f:
        grp = f[f"data/{demo_key}"]
        eef_pose = np.asarray(grp["datagen_info/eef_pose"])
        base_pose = np.asarray(grp["datagen_info/base_pose"])
        object_poses = {}
        if "datagen_info/object_poses" in grp:
            for name in grp["datagen_info/object_poses"].keys():
                object_poses[name] = np.asarray(grp[f"datagen_info/object_poses/{name}"])
        out = {
            "horizon": int(eef_pose.shape[0]),
            "left_eef": eef_pose[:, 0:4, :][:, :3, 3],
            "right_eef": eef_pose[:, 4:8, :][:, :3, 3],
            "base": base_pose[:, :3, 3],
            "base_yaw": np.unwrap(np.arctan2(base_pose[:, 1, 0], base_pose[:, 0, 0])),
            "objects": {name: pose[:, :3, 3] for name, pose in object_poses.items()},
            "left_mp_ranges": np.asarray(grp["left_mp_ranges"]).astype(int).tolist()
            if "left_mp_ranges" in grp
            else [],
            "right_mp_ranges": np.asarray(grp["right_mp_ranges"]).astype(int).tolist()
            if "right_mp_ranges" in grp
            else [],
            "subtask_lengths": np.asarray(grp["subtask_lengths"]).astype(int).tolist()
            if "subtask_lengths" in grp
            else [],
        }
        return out


def _phase_ranges(subtask_lengths, horizon):
    ranges = []
    start = 0
    for idx, length in enumerate(subtask_lengths, start=1):
        end = min(horizon, start + int(length))
        ranges.append((f"phase{idx}", start, end))
        start = end
    if not ranges:
        ranges.append(("episode", 0, horizon))
    if ranges[-1][2] < horizon:
        ranges.append(("tail", ranges[-1][2], horizon))
    return ranges


def _load_phase_logs(path):
    if path is None:
        return {}
    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"episode log not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    phase_logs = data.get("phase_logs")
    if isinstance(phase_logs, list) and phase_logs:
        phase_logs = phase_logs[0]
    return phase_logs if isinstance(phase_logs, dict) else {}


def _count_prealign_steps(phase_log):
    count = 0
    for record in phase_log.get("toggle_marker_contact_prealign", []):
        if not isinstance(record, dict):
            continue
        if record.get("stage") == "exec_done" and isinstance(record.get("q_traj_len"), int):
            return int(record["q_traj_len"])
        if record.get("stage") == "exec_step" and isinstance(record.get("q_idx"), int):
            count = max(count, int(record["q_idx"]) + 1)
    return count


def _count_post_press_steps(phase_log):
    count = 0
    for record in phase_log.get("toggle_marker_post_mp_press", []):
        if not isinstance(record, dict):
            continue
        press_step = record.get("press_step")
        if isinstance(press_step, int):
            count = max(count, press_step + 1)
    return count


def _add_execution_subsegment_ranges(ranges, phase_ranges, left_mp_ranges, right_mp_ranges, phase_logs):
    if not phase_logs:
        return ranges, []
    added = []
    mp_ranges = _range_union(left_mp_ranges or right_mp_ranges)
    for idx, (_, mp_end) in enumerate(mp_ranges, start=1):
        if idx > len(phase_ranges):
            continue
        _, _, phase_end = phase_ranges[idx - 1]
        if mp_end >= phase_end:
            continue

        # HDF5 phase labels are 1-based. episode_logs phase keys are execution-phase
        # indices and are 0-based for the same generated subtask.
        log_key = str(idx - 1)
        phase_log = phase_logs.get(log_key, {})
        if not isinstance(phase_log, dict):
            continue

        visibility = phase_log.get("visibility_stats") if isinstance(phase_log.get("visibility_stats"), dict) else {}
        arm_replay_steps = int(visibility.get("arm_replay_steps") or 0)
        prealign_steps = _count_prealign_steps(phase_log)
        post_press_steps = _count_post_press_steps(phase_log)

        cursor = int(mp_end)
        specs = [
            ("arm_replay", arm_replay_steps),
            ("contact_prealign", prealign_steps),
            ("post_press", post_press_steps),
        ]
        for name, length in specs:
            if length <= 0 or cursor >= phase_end:
                continue
            end = min(phase_end, cursor + int(length))
            label = f"phase{idx}_after_mp_{name}"
            ranges.append((label, cursor, end))
            added.append(
                {
                    "label": label,
                    "phase_label": f"phase{idx}",
                    "episode_log_phase": log_key,
                    "source": name,
                    "start": int(cursor),
                    "end": int(end),
                    "requested_length": int(length),
                }
            )
            cursor = end
        if cursor < phase_end:
            label = f"phase{idx}_after_mp_unattributed"
            ranges.append((label, cursor, phase_end))
            added.append(
                {
                    "label": label,
                    "phase_label": f"phase{idx}",
                    "episode_log_phase": log_key,
                    "source": "unattributed",
                    "start": int(cursor),
                    "end": int(phase_end),
                    "requested_length": int(phase_end - cursor),
                }
            )
    return ranges, added


def _range_union(mp_ranges):
    return [(int(a), int(b)) for a, b in mp_ranges]


def _add_mp_and_replay_ranges(phase_ranges, left_mp_ranges, right_mp_ranges):
    ranges = list(phase_ranges)
    mp_ranges = _range_union(left_mp_ranges or right_mp_ranges)
    for idx, (a, b) in enumerate(mp_ranges, start=1):
        ranges.append((f"phase{idx}_mp", a, b))
        if idx <= len(phase_ranges):
            _, phase_start, phase_end = phase_ranges[idx - 1]
            if b < phase_end:
                ranges.append((f"phase{idx}_after_mp", b, phase_end))
    return ranges


def _path_metrics(pos, start, end):
    seg = np.asarray(pos[start:end], dtype=float)
    if len(seg) == 0:
        return None
    if len(seg) == 1:
        diffs = np.zeros((0, seg.shape[-1]))
    else:
        diffs = np.diff(seg, axis=0)
    step = np.linalg.norm(diffs, axis=1) if len(diffs) else np.asarray([], dtype=float)
    path = float(step.sum())
    net = float(np.linalg.norm(seg[-1] - seg[0]))
    return {
        "start_step": int(start),
        "end_step": int(end),
        "num_steps": int(end - start),
        "path_m": path,
        "net_m": net,
        "path_to_net_ratio": float(path / max(net, 1e-9)),
        "max_step_m": float(step.max()) if len(step) else 0.0,
        "mean_step_m": float(step.mean()) if len(step) else 0.0,
        "start_pos": seg[0].tolist(),
        "end_pos": seg[-1].tolist(),
        "min_pos": seg.min(axis=0).tolist(),
        "max_pos": seg.max(axis=0).tolist(),
    }


def _yaw_metrics(yaw, start, end):
    seg = np.asarray(yaw[start:end], dtype=float)
    if len(seg) == 0:
        return None
    step = np.abs(np.diff(seg)) if len(seg) > 1 else np.asarray([], dtype=float)
    path = float(step.sum())
    net = float(abs(seg[-1] - seg[0]))
    return {
        "start_step": int(start),
        "end_step": int(end),
        "num_steps": int(end - start),
        "path_rad": path,
        "net_rad": net,
        "path_to_net_ratio": float(path / max(net, 1e-9)),
        "max_step_rad": float(step.max()) if len(step) else 0.0,
        "start_yaw": float(seg[0]),
        "end_yaw": float(seg[-1]),
    }


def _compute_metrics(data, ranges, object_names):
    rows = []
    metrics = {"horizon": data["horizon"], "ranges": [], "left_mp_ranges": data["left_mp_ranges"], "right_mp_ranges": data["right_mp_ranges"]}
    tracked = {
        "base_xy": data["base"][:, :2],
        "left_eef": data["left_eef"],
        "right_eef": data["right_eef"],
    }
    for obj_name in object_names:
        if obj_name in data["objects"]:
            tracked[obj_name] = data["objects"][obj_name]

    for label, start, end in ranges:
        entry = {"label": label, "start": int(start), "end": int(end), "tracks": {}}
        for track_name, pos in tracked.items():
            m = _path_metrics(pos, start, end)
            if m is None:
                continue
            entry["tracks"][track_name] = m
            rows.append({"range": label, "track": track_name, **m})
        ym = _yaw_metrics(data["base_yaw"], start, end)
        if ym is not None:
            entry["tracks"]["base_yaw"] = ym
            rows.append({"range": label, "track": "base_yaw", **ym})
        metrics["ranges"].append(entry)
    return metrics, rows


def _plot_topdown(data, phase_ranges, out_path, object_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"left_eef": "#1f77b4", "right_eef": "#ff7f0e", "base": "#2ca02c", "radio_89": "#d62728"}
    tracks = {
        "left_eef": data["left_eef"][:, :2],
        "right_eef": data["right_eef"][:, :2],
        "base": data["base"][:, :2],
    }
    for obj_name in object_names:
        if obj_name in data["objects"]:
            tracks[obj_name] = data["objects"][obj_name][:, :2]

    for name, pos in tracks.items():
        color = colors.get(name, None)
        ax.plot(pos[:, 0], pos[:, 1], label=name, linewidth=1.8, color=color)
        ax.scatter(pos[0, 0], pos[0, 1], marker="o", s=40, color=color)
        ax.scatter(pos[-1, 0], pos[-1, 1], marker="x", s=60, color=color)

    for phase_name, start, end in phase_ranges:
        p = data["base"][start, :2]
        ax.text(p[0], p[1], f"{phase_name} start {start}", fontsize=8)
        if end > start:
            q = data["base"][end - 1, :2]
            ax.text(q[0], q[1], f"{phase_name} end {end}", fontsize=8)

    ax.set_title("Generated trajectory top-down view")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_timeseries(data, phase_ranges, out_path, object_names):
    t = np.arange(data["horizon"])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    tracks = {
        "base_xy_step": np.r_[0.0, np.linalg.norm(np.diff(data["base"][:, :2], axis=0), axis=1)],
        "left_eef_step": np.r_[0.0, np.linalg.norm(np.diff(data["left_eef"], axis=0), axis=1)],
        "right_eef_step": np.r_[0.0, np.linalg.norm(np.diff(data["right_eef"], axis=0), axis=1)],
    }
    if "radio_89" in data["objects"]:
        tracks["radio_step"] = np.r_[0.0, np.linalg.norm(np.diff(data["objects"]["radio_89"], axis=0), axis=1)]

    for name, vals in tracks.items():
        axes[0].plot(t, vals, label=name, linewidth=1.0)
    axes[0].set_ylabel("step dist (m)")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, data["base"][:, 0], label="base x")
    axes[1].plot(t, data["base"][:, 1], label="base y")
    axes[1].plot(t, data["base_yaw"], label="base yaw")
    axes[1].set_ylabel("base")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, data["left_eef"][:, 2], label="left z")
    axes[2].plot(t, data["right_eef"][:, 2], label="right z")
    if "radio_89" in data["objects"]:
        axes[2].plot(t, data["objects"]["radio_89"][:, 2], label="radio z")
    axes[2].set_ylabel("z (m)")
    axes[2].legend(loc="upper right")

    if "radio_89" in data["objects"]:
        rel_left = np.linalg.norm(data["left_eef"] - data["objects"]["radio_89"], axis=1)
        rel_right = np.linalg.norm(data["right_eef"] - data["objects"]["radio_89"], axis=1)
        axes[3].plot(t, rel_left, label="left-radio")
        axes[3].plot(t, rel_right, label="right-radio")
    axes[3].set_ylabel("dist (m)")
    axes[3].set_xlabel("generated step")
    axes[3].legend(loc="upper right")

    for ax in axes:
        for phase_name, start, end in phase_ranges:
            ax.axvspan(start, end, alpha=0.08)
            ax.axvline(start, color="k", linewidth=0.5, alpha=0.35)
            ax.text(start, ax.get_ylim()[1], phase_name, rotation=90, va="top", fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Generated trajectory time series")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to generated demo.hdf5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-log", help="Optional episode_logs.json for replay/prealign/press subsegments")
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_names = [x for x in args.objects.split(",") if x]

    data = _load_demo(dataset_path, demo_key=args.demo_key)
    phase_ranges = _phase_ranges(data["subtask_lengths"], data["horizon"])
    ranges = _add_mp_and_replay_ranges(phase_ranges, data["left_mp_ranges"], data["right_mp_ranges"])
    phase_logs = _load_phase_logs(args.episode_log)
    ranges, execution_subsegments = _add_execution_subsegment_ranges(
        ranges,
        phase_ranges,
        data["left_mp_ranges"],
        data["right_mp_ranges"],
        phase_logs,
    )
    metrics, rows = _compute_metrics(data, ranges, object_names)
    metrics.update(
        {
            "dataset": str(dataset_path),
            "phase_ranges": [{"label": label, "start": start, "end": end} for label, start, end in phase_ranges],
            "execution_subsegments": execution_subsegments,
            "tracked_objects": [name for name in object_names if name in data["objects"]],
        }
    )

    (output_dir / "trajectory_quality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_csv(output_dir / "trajectory_quality_metrics.csv", rows)
    _plot_topdown(data, phase_ranges, output_dir / "trajectory_topdown.png", object_names)
    _plot_timeseries(data, phase_ranges, output_dir / "trajectory_timeseries.png", object_names)
    print(json.dumps({"output_dir": str(output_dir), "metrics": str(output_dir / "trajectory_quality_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
