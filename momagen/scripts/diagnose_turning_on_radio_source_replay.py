"""
Replay a source turning_on_radio demo and record toggle-trigger geometry.

This is an experiment diagnostic, not a data generation path. It reuses the
OmniGibson DataPlaybackWrapper so the source HDF5 is restored step-by-step,
then writes a compact JSON summary of the real ToggledOn predicate signals.
"""

import argparse
import json
import os
import tempfile

import h5py
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.object_states import ToggledOn


DEFAULT_DATASET = (
    "/home/ubuntu/repo/MoMaGen/momagen/datasets/processed_source_demos/"
    "r1_turning_on_radio_raw_episode_00000010.hdf5"
)
DEFAULT_OUTPUT = "/tmp/momagen_turning_on_radio_round60/A69_source_replay_trigger_diag.json"
DEFAULT_RADII = (0.02235804684460163, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15)


def _to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value.tolist()


def _as_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return float(arr)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


class ToggleReplayDiagnosticWrapper(DataPlaybackWrapper):
    def __init__(self, *args, radii=None, max_hit_records=20, **kwargs):
        self.records = []
        self.radii = tuple(radii or DEFAULT_RADII)
        self.max_hit_records = int(max_hit_records)
        self._input_actions_np = None
        self._input_gripper_actions_np = None
        self._input_eef_pose_np = None
        self._step_index = -1
        super().__init__(*args, **kwargs)

    def _process_obs(self, obs, info):
        del obs, info
        return {}

    def playback_episode(self, episode_id, record_data=True, video_writers=None):
        traj_grp = self.input_hdf5["data"][f"demo_{episode_id}"]
        self._input_actions_np = np.asarray(traj_grp["action"])
        datagen_info = traj_grp.get("datagen_info", None)
        if datagen_info is not None:
            if "gripper_action" in datagen_info:
                self._input_gripper_actions_np = np.asarray(datagen_info["gripper_action"])
            if "eef_pose" in datagen_info:
                self._input_eef_pose_np = np.asarray(datagen_info["eef_pose"])
        self._step_index = -1
        return super().playback_episode(
            episode_id=episode_id,
            record_data=record_data,
            video_writers=video_writers,
        )

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        self._step_index += 1
        record = self._snapshot(action=action, reward=reward, terminated=terminated, truncated=truncated)
        self.records.append(record)
        return super()._parse_step_data(
            action=action,
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _snapshot(self, action, reward, terminated, truncated):
        step = self._step_index
        record = {
            "step": int(step),
            "reward": _as_float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_success": self._task_success(),
            "input_action": _to_list(action),
            "input_gripper_action": self._dataset_row(self._input_gripper_actions_np, step),
            "input_eef_pose": self._dataset_row(self._input_eef_pose_np, step),
            "objects": {},
        }
        robots = list(getattr(self, "robots", []) or [])
        if robots:
            robot = robots[0]
            record["robot"] = self._robot_snapshot(robot)
        else:
            robot = None
            record["robot"] = {"error": "no_robot"}

        for obj in getattr(self.scene, "objects", []):
            if ToggledOn not in getattr(obj, "states", {}):
                continue
            record["objects"][obj.name] = self._toggle_snapshot(obj, robot)

        return record

    @staticmethod
    def _dataset_row(array, step):
        if array is None or step < 0 or step >= len(array):
            return None
        return _to_list(array[step])

    def _task_success(self):
        try:
            return bool(self.task.success)
        except Exception:
            try:
                success = self.task.get_reward_termination(self)[2]
                return bool(success)
            except Exception as exc:
                return f"ERR:{type(exc).__name__}: {exc}"

    def _robot_snapshot(self, robot):
        snapshot = {
            "name": getattr(robot, "name", None),
            "eef_link_names": dict(getattr(robot, "eef_link_names", {}) or {}),
            "finger_links": {},
            "eef_poses": {},
        }
        for arm, link in (getattr(robot, "eef_links", {}) or {}).items():
            try:
                pos, quat = link.get_position_orientation()
                snapshot["eef_poses"][arm] = {
                    "link": getattr(link, "name", str(link)),
                    "prim_path": getattr(link, "prim_path", None),
                    "pos": _to_list(pos),
                    "quat": _to_list(quat),
                }
            except Exception as exc:
                snapshot["eef_poses"][arm] = f"ERR:{type(exc).__name__}: {exc}"
        for arm, links in (getattr(robot, "finger_links", {}) or {}).items():
            snapshot["finger_links"][arm] = []
            for link in links:
                try:
                    pos, quat = link.get_position_orientation()
                    snapshot["finger_links"][arm].append(
                        {
                            "link": getattr(link, "name", str(link)),
                            "body_name": getattr(link, "body_name", None),
                            "prim_path": getattr(link, "prim_path", None),
                            "pos": _to_list(pos),
                            "quat": _to_list(quat),
                        }
                    )
                except Exception as exc:
                    snapshot["finger_links"][arm].append(f"ERR:{type(exc).__name__}: {exc}")
        return snapshot

    def _toggle_snapshot(self, obj, robot):
        state = obj.states[ToggledOn]
        marker = getattr(state, "visual_marker", None)
        marker_pos = None
        marker_quat = None
        marker_radius = None
        if marker is not None:
            marker_pos_raw, marker_quat_raw = marker.get_position_orientation()
            marker_pos = np.asarray(_to_list(marker_pos_raw), dtype=np.float64)
            marker_quat = np.asarray(_to_list(marker_quat_raw), dtype=np.float64)
            try:
                extent = np.asarray(_to_list(getattr(marker, "extent", None)), dtype=np.float64)
                scale = np.asarray(_to_list(getattr(marker, "scale", None)), dtype=np.float64)
                marker_radius = float(np.min(extent * scale))
            except Exception as exc:
                marker_radius = f"ERR:{type(exc).__name__}: {exc}"

        finger_contact_objs = getattr(ToggledOn, "_finger_contact_objs", None)
        try:
            obj_in_finger_contact_objs = None if finger_contact_objs is None else obj in finger_contact_objs
        except Exception as exc:
            obj_in_finger_contact_objs = f"ERR:{type(exc).__name__}: {exc}"

        snapshot = {
            "value": bool(state.get_value()),
            "robot_can_toggle_steps": int(getattr(state, "robot_can_toggle_steps", -1)),
            "obj_in_finger_contact_objs": obj_in_finger_contact_objs,
            "marker": {
                "pos": _to_list(marker_pos),
                "quat": _to_list(marker_quat),
                "radius": marker_radius,
                "prim_path": getattr(marker, "prim_path", None) if marker is not None else None,
            },
            "eef_dist_to_marker": {},
            "finger_dists_to_marker": {},
            "overlap_sphere_probe": {},
        }

        if robot is not None and marker_pos is not None:
            for arm, link in (getattr(robot, "eef_links", {}) or {}).items():
                try:
                    eef_pos = _to_list(link.get_position_orientation()[0])
                    snapshot["eef_dist_to_marker"][arm] = _dist(eef_pos, marker_pos)
                except Exception as exc:
                    snapshot["eef_dist_to_marker"][arm] = f"ERR:{type(exc).__name__}: {exc}"
            for arm, links in (getattr(robot, "finger_links", {}) or {}).items():
                rows = []
                for link in links:
                    try:
                        finger_pos = _to_list(link.get_position_orientation()[0])
                        rows.append(
                            {
                                "link": getattr(link, "name", str(link)),
                                "body_name": getattr(link, "body_name", None),
                                "prim_path": getattr(link, "prim_path", None),
                                "pos": finger_pos,
                                "dist": _dist(finger_pos, marker_pos),
                            }
                        )
                    except Exception as exc:
                        rows.append({"error": f"{type(exc).__name__}: {exc}"})
                rows.sort(key=lambda row: row.get("dist", float("inf")))
                snapshot["finger_dists_to_marker"][arm] = rows
            snapshot["overlap_sphere_probe"] = self._overlap_probe(robot, marker_pos)

        return snapshot

    def _overlap_probe(self, robot, marker_pos):
        finger_paths = {
            getattr(link, "prim_path", None)
            for links in (getattr(robot, "finger_links", {}) or {}).values()
            for link in links
        }
        finger_paths.discard(None)
        probes = []
        for radius in self.radii:
            hits = []
            valid_hit = False

            def _report(hit):
                nonlocal valid_hit
                rigid_body = str(getattr(hit, "rigid_body", ""))
                is_finger = rigid_body in finger_paths
                valid_hit = valid_hit or is_finger
                if len(hits) < self.max_hit_records:
                    hits.append({"rigid_body": rigid_body, "is_robot_finger": is_finger})
                return True

            try:
                og.sim.psqi.overlap_sphere(radius=float(radius), pos=marker_pos.tolist(), reportFn=_report)
                probes.append(
                    {
                        "radius": float(radius),
                        "valid_robot_finger_hit": bool(valid_hit),
                        "num_recorded_hits": len(hits),
                        "hits": hits,
                    }
                )
            except Exception as exc:
                probes.append({"radius": float(radius), "error": f"{type(exc).__name__}: {exc}"})

        first_finger_hit_radius = None
        for probe in probes:
            if probe.get("valid_robot_finger_hit"):
                first_finger_hit_radius = probe["radius"]
                break
        return {
            "finger_paths": sorted(finger_paths),
            "probes": probes,
            "first_finger_hit_radius": first_finger_hit_radius,
        }


def _summarize(records):
    summary = {
        "num_records": len(records),
        "first_task_success_step": None,
        "first_toggle_value_step": None,
        "first_can_toggle_step": None,
        "first_obj_contact_step": None,
        "first_overlap_by_radius": {},
        "max_robot_can_toggle_steps": 0,
        "best_finger_dist": None,
        "nearest_finger_event": None,
        "interesting_steps": [],
    }

    best_dist = float("inf")
    interesting = set()
    for record in records:
        step = record["step"]
        if record.get("task_success") is True and summary["first_task_success_step"] is None:
            summary["first_task_success_step"] = step
            interesting.add(step)
        for obj_name, obj_record in record.get("objects", {}).items():
            can_steps = int(obj_record.get("robot_can_toggle_steps", 0))
            summary["max_robot_can_toggle_steps"] = max(summary["max_robot_can_toggle_steps"], can_steps)
            if obj_record.get("value") and summary["first_toggle_value_step"] is None:
                summary["first_toggle_value_step"] = step
                interesting.add(step)
            if can_steps > 0 and summary["first_can_toggle_step"] is None:
                summary["first_can_toggle_step"] = step
                interesting.add(step)
            if obj_record.get("obj_in_finger_contact_objs") is True and summary["first_obj_contact_step"] is None:
                summary["first_obj_contact_step"] = step
                interesting.add(step)

            for arm, rows in obj_record.get("finger_dists_to_marker", {}).items():
                if not rows or "dist" not in rows[0]:
                    continue
                if rows[0]["dist"] < best_dist:
                    best_dist = rows[0]["dist"]
                    summary["nearest_finger_event"] = {
                        "step": step,
                        "object": obj_name,
                        "arm": arm,
                        "finger": rows[0],
                        "robot_can_toggle_steps": can_steps,
                        "obj_in_finger_contact_objs": obj_record.get("obj_in_finger_contact_objs"),
                    }
            for probe in obj_record.get("overlap_sphere_probe", {}).get("probes", []):
                radius_key = str(probe.get("radius"))
                if probe.get("valid_robot_finger_hit") and radius_key not in summary["first_overlap_by_radius"]:
                    summary["first_overlap_by_radius"][radius_key] = {
                        "step": step,
                        "object": obj_name,
                        "radius": probe.get("radius"),
                    }
                    interesting.add(step)

    if best_dist < float("inf"):
        summary["best_finger_dist"] = best_dist

    for event in (
        summary["first_task_success_step"],
        summary["first_toggle_value_step"],
        summary["first_can_toggle_step"],
        summary["first_obj_contact_step"],
    ):
        if event is not None:
            interesting.update(range(max(0, event - 3), min(len(records), event + 8)))
    if summary["nearest_finger_event"] is not None:
        step = summary["nearest_finger_event"]["step"]
        interesting.update(range(max(0, step - 3), min(len(records), step + 8)))
    summary["interesting_steps"] = sorted(interesting)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--radii", default=",".join(str(radius) for radius in DEFAULT_RADII))
    parser.add_argument("--max-hit-records", type=int, default=20)
    parser.add_argument("--record-step-snapshots", action="store_true")
    args = parser.parse_args()

    gm.ENABLE_TRANSITION_RULES = False
    radii = tuple(float(part) for part in args.radii.split(",") if part.strip())

    with h5py.File(args.dataset, "r") as f:
        num_samples = int(f[f"data/demo_{args.episode_id}"].attrs["num_samples"])
        source_attrs = {key: f["data"].attrs[key] for key in f["data"].attrs}

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
            n_render_iterations=1,
            only_successes=False,
            include_contacts=True,
        )
        env.radii = radii
        env.max_hit_records = args.max_hit_records
        env.playback_episode(episode_id=args.episode_id, record_data=True)
        records = env.records
        summary = _summarize(records)
        payload = {
            "dataset": args.dataset,
            "episode_id": args.episode_id,
            "num_samples": num_samples,
            "source_attrs": {key: str(value)[:500] for key, value in source_attrs.items()},
            "radii": list(radii),
            "summary": summary,
            "records": records if args.record_step_snapshots else [records[i] for i in summary["interesting_steps"]],
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
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
