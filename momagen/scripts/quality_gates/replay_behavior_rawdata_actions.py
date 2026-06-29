#!/usr/bin/env python3
"""Replay a BEHAVIOR rawdata action window from a single restored state.

This diagnostic differs from DataPlaybackWrapper playback: it restores only one
rawdata state, then steps the recorded actions open-loop. That makes it a closer
positive control for eval_segment action replay.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm

from momagen.scripts.diagnose_turning_on_radio_source_replay import (
    ToggleReplayDiagnosticWrapper,
    _summarize,
)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    Path(tmp_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--n-render-iterations", type=int, default=1)
    parser.add_argument("--include-contacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-step-snapshots", action="store_true")
    args = parser.parse_args()

    gm.ENABLE_TRANSITION_RULES = False

    with h5py.File(args.dataset, "r") as f:
        grp = f["data"][f"demo_{args.episode_id}"]
        total_actions = int(len(grp["action"]))
        total_states = int(len(grp["state"]))
        if args.start < 0 or args.start >= total_states:
            raise ValueError(f"start={args.start} outside state range [0, {total_states})")
        if args.end <= args.start or args.end > total_actions:
            raise ValueError(f"end={args.end} outside action range ({args.start}, {total_actions}]")

    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    tmp_path = tmp.name
    tmp.close()

    env = None
    try:
        env = ToggleReplayDiagnosticWrapper.create_from_hdf5(
            input_path=args.dataset,
            output_path=tmp_path,
            robot_obs_modalities=(),
            robot_sensor_config=None,
            external_sensors_config=None,
            n_render_iterations=args.n_render_iterations,
            only_successes=False,
            include_contacts=args.include_contacts,
        )
        data_grp = env.input_hdf5["data"][f"demo_{args.episode_id}"]
        state = data_grp["state"]
        state_size = data_grp["state_size"]
        action = data_grp["action"]

        env.scene.restore(env.scene_file, update_initial_file=True)
        env.reset()
        og.sim.load_state(th.as_tensor(state[args.start, : int(state_size[args.start])]), serialized=True)

        records = []
        last_i = args.start
        for i in range(args.start, args.end):
            last_i = i + 1
            if i == args.start or ((i - args.start) % max(1, args.snapshot_every) == 0):
                records.append(env._snapshot(action=action[i], reward=0.0, terminated=False, truncated=False))
                records[-1]["step"] = int(i)
                records[-1]["relative_step"] = int(i - args.start)
            env.current_obs, _, terminated, truncated, _ = env.env.step(
                action=action[i],
                n_render_iterations=args.n_render_iterations,
            )
            if terminated or truncated:
                break

        final_step = min(args.end, last_i)
        records.append(env._snapshot(action=np.zeros_like(action[args.start]), reward=0.0, terminated=False, truncated=False))
        records[-1]["step"] = int(final_step)
        records[-1]["relative_step"] = int(final_step - args.start)
        summary = _summarize(records)
        records_by_step = {record["step"]: record for record in records}
        payload = {
            "dataset": args.dataset,
            "episode_id": int(args.episode_id),
            "start": int(args.start),
            "end": int(args.end),
            "total_actions": total_actions,
            "total_states": total_states,
            "include_contacts": bool(args.include_contacts),
            "n_render_iterations": int(args.n_render_iterations),
            "summary": summary,
            "records": records
            if args.record_step_snapshots
            else [records_by_step[i] for i in summary["interesting_steps"] if i in records_by_step],
        }
        _write_json(args.output, payload)
        print(json.dumps(summary, indent=2))
        print(f"Wrote {args.output}")
    finally:
        try:
            if env is not None:
                env.input_hdf5.close()
                if getattr(env, "hdf5_file", None) is not None:
                    env.hdf5_file.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            og.shutdown()


if __name__ == "__main__":
    main()
