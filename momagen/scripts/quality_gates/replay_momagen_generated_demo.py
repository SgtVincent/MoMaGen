#!/usr/bin/env python3
"""Replay a MoMaGen generated HDF5 by restoring states[0] and stepping actions.

This is intentionally separate from BEHAVIOR DataPlaybackWrapper because MoMaGen
outputs `actions` / `states` / `env_args`, not DataCollectionWrapper's
`action` / `state` / `state_size` / `scene_file` schema.
"""
import argparse, json, os, sys, tempfile, traceback
from pathlib import Path
from copy import deepcopy

import h5py
import imageio.v2 as imageio
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import ToggledOn
import omnigibson.utils.transform_utils as T

import momagen.utils.robomimic_utils as RobomimicUtils


def _to_list(x):
    if x is None: return None
    if hasattr(x, 'detach'): x=x.detach().cpu().numpy()
    return np.asarray(x).tolist()


def _dist(a,b):
    return float(np.linalg.norm(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)))


def _task_success(env):
    try:
        s=env.is_success()
        return {k: bool(v) for k,v in s.items()}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def _state_error(ref_state):
    try:
        cur=_to_list(og.sim.dump_state(serialized=True))
        ref=np.asarray(ref_state, dtype=float)
        cur=np.asarray(cur, dtype=float)
        if cur.shape != ref.shape:
            return {'shape_match': False, 'current_shape': list(cur.shape), 'ref_shape': list(ref.shape)}
        delta=cur-ref
        return {
            'shape_match': True,
            'l2': float(np.linalg.norm(delta)),
            'mean_abs': float(np.mean(np.abs(delta))),
            'max_abs': float(np.max(np.abs(delta))),
        }
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def _snapshot(env, step, ref_state=None, target_objects=None):
    out={'step': int(step), 'success': _task_success(env), 'objects': {}}
    if ref_state is not None:
        out['state_error_vs_hdf5']=_state_error(ref_state)
    og_env=getattr(env,'env',env)
    robot=og_env.robots[0] if getattr(og_env,'robots',None) else None
    target_objects=set(target_objects or [])
    for obj in getattr(og_env.scene,'objects',[]):
        if target_objects and obj.name not in target_objects:
            continue
        if ToggledOn not in getattr(obj,'states',{}):
            continue
        state=obj.states[ToggledOn]
        marker=getattr(state,'visual_marker',None)
        marker_pos=None; marker_radius=None
        if marker is not None:
            marker_pos=np.asarray(_to_list(marker.get_position_orientation()[0]), dtype=float)
            try:
                marker_radius=float(np.min(np.asarray(_to_list(marker.extent),dtype=float)*np.asarray(_to_list(marker.scale),dtype=float)))
            except Exception as e:
                marker_radius=f'ERR:{type(e).__name__}: {e}'
        finger_contact_objs=getattr(ToggledOn,'_finger_contact_objs',None)
        try:
            obj_in_contact=None if finger_contact_objs is None else obj in finger_contact_objs
        except Exception as e:
            obj_in_contact=f'ERR:{type(e).__name__}: {e}'
        row={
            'value': bool(state.get_value()),
            'robot_can_toggle_steps': int(getattr(state,'robot_can_toggle_steps',-1)),
            'obj_in_finger_contact_objs': obj_in_contact,
            'marker_pos': _to_list(marker_pos),
            'marker_radius': marker_radius,
            'finger_min_dist_to_marker': {},
            'overlap_primary_hit': None,
            'overlap_first_hit_radius': None,
            'overlap_probes': [],
        }
        if robot is not None and marker_pos is not None:
            finger_paths={getattr(link,'prim_path',None) for links in getattr(robot,'finger_links',{}).values() for link in links}
            finger_paths.discard(None)
            row['finger_paths'] = sorted(finger_paths)
            for arm,links in getattr(robot,'finger_links',{}).items():
                vals=[]
                for link in links:
                    try:
                        vals.append({'link': getattr(link,'name',str(link)), 'prim_path': getattr(link,'prim_path',None), 'dist': _dist(_to_list(link.get_position_orientation()[0]), marker_pos)})
                    except Exception as e:
                        vals.append({'error': f'{type(e).__name__}: {e}'})
                vals=[v for v in vals if 'dist' in v]
                row['finger_min_dist_to_marker'][arm]=min(vals,key=lambda x:x['dist']) if vals else None
            radii=[marker_radius,0.03,0.05,0.075,0.10,0.125,0.15] if isinstance(marker_radius,float) else [0.03,0.05,0.075,0.10,0.125,0.15]
            seen=[]
            for radius in radii:
                if radius in seen: continue
                seen.append(radius)
                valid=False; hits=[]
                num_hits=0
                def report(hit):
                    nonlocal valid, num_hits
                    num_hits += 1
                    rb=str(getattr(hit,'rigid_body',''))
                    is_finger=rb in finger_paths
                    valid = valid or is_finger
                    if len(hits)<64: hits.append({'rigid_body':rb,'is_robot_finger':is_finger})
                    return True
                try:
                    og.sim.psqi.overlap_sphere(radius=float(radius), pos=marker_pos.tolist(), reportFn=report)
                    row['overlap_probes'].append({
                        'radius': float(radius),
                        'valid_robot_finger_hit': bool(valid),
                        'num_hits': int(num_hits),
                        'hits': hits,
                    })
                    if radius == marker_radius:
                        row['overlap_primary_hit']=bool(valid)
                    if valid and row['overlap_first_hit_radius'] is None:
                        row['overlap_first_hit_radius']=float(radius)
                except Exception as e:
                    row.setdefault('overlap_errors',[]).append(f'{radius}:{type(e).__name__}: {e}')
        out['objects'][obj.name]=row
    return out


def summarize(records):
    summary={'num_records': len(records), 'first_task_success_step': None, 'first_toggle_value_step': None, 'first_can_toggle_step': None, 'first_primary_overlap_step': None, 'max_robot_can_toggle_steps': 0, 'best_left_finger_dist': None, 'state_error': {'max_abs_max': None, 'mean_abs_max': None, 'l2_max': None, 'shape_mismatch_steps': []}}
    best=float('inf')
    for r in records:
        step=r['step']
        if r.get('success',{}).get('task') is True and summary['first_task_success_step'] is None:
            summary['first_task_success_step']=step
        err=r.get('state_error_vs_hdf5')
        if isinstance(err,dict):
            if err.get('shape_match') is False:
                summary['state_error']['shape_mismatch_steps'].append(step)
            for key in ('max_abs','mean_abs','l2'):
                if isinstance(err.get(key),(int,float)):
                    out_key=f'{key}_max'
                    prev=summary['state_error'].get(out_key)
                    summary['state_error'][out_key]=float(err[key]) if prev is None else max(float(prev), float(err[key]))
        for obj,row in r.get('objects',{}).items():
            can=row.get('robot_can_toggle_steps') or 0
            summary['max_robot_can_toggle_steps']=max(summary['max_robot_can_toggle_steps'], can)
            if row.get('value') and summary['first_toggle_value_step'] is None:
                summary['first_toggle_value_step']=step
            if can>0 and summary['first_can_toggle_step'] is None:
                summary['first_can_toggle_step']=step
            if row.get('overlap_primary_hit') and summary['first_primary_overlap_step'] is None:
                summary['first_primary_overlap_step']=step
            left=(row.get('finger_min_dist_to_marker') or {}).get('left')
            if isinstance(left,dict) and isinstance(left.get('dist'),(int,float)) and left['dist']<best:
                best=left['dist']; summary['best_left_finger_dist']={'step':step,'object':obj,'dist':left['dist'],'link':left.get('link')}
    return summary


def write_payload(path, payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp_path=f"{path}.tmp"
    Path(tmp_path).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    os.replace(tmp_path,path)


def _camera_quat_looking_at(cam_pos, target):
    forward = np.asarray(target, dtype=np.float32) - np.asarray(cam_pos, dtype=np.float32)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    corrected_up = np.cross(right, forward)
    corrected_up = corrected_up / max(float(np.linalg.norm(corrected_up)), 1e-6)
    # USD camera looks along local -Z; columns are local axes in world frame.
    rot = th.tensor(np.stack([right, corrected_up, -forward], axis=1), dtype=th.float32)
    return T.mat2quat(rot)


def _set_review_camera(env, target_objects=None):
    if not target_objects:
        return
    og_env = getattr(env, 'env', env)
    target_obj = None
    target_names = set(target_objects)
    for obj in getattr(og_env.scene, 'objects', []):
        if obj.name in target_names:
            target_obj = obj
            break
    if target_obj is None:
        return
    target_pos = _to_list(target_obj.get_position_orientation()[0])
    target = np.asarray(target_pos, dtype=np.float32) + np.asarray([0.0, 0.0, 0.25], dtype=np.float32)
    cam_pos = target + np.asarray([-1.15, -1.25, 0.85], dtype=np.float32)
    quat = _camera_quat_looking_at(cam_pos, target)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(cam_pos, dtype=th.float32),
        orientation=quat,
    )


def _capture_frame(env, target_objects=None):
    _set_review_camera(env, target_objects=target_objects)
    for _ in range(3):
        og.sim.render()
    frame = og.sim.viewer_camera.get_obs()[0]['rgb']
    if hasattr(frame, 'detach'):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if frame.max(initial=0) <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[..., :3]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--snapshot-every', type=int, default=1)
    ap.add_argument('--init-curobo', action='store_true')
    ap.add_argument('--call-wrapper-reset', action='store_true', help='Call EnvOmniGibson.reset() before loading states[0]. Disabled by default because replay restores the exact HDF5 initial state.')
    ap.add_argument('--target-object', action='append', default=None, help='Only record diagnostics for this object name. Can be repeated.')
    ap.add_argument('--video-output', default=None, help='Optional MP4 path for rendered replay video.')
    ap.add_argument('--video-every', type=int, default=1, help='Append one video frame every N replay steps.')
    ap.add_argument('--video-fps', type=int, default=20)
    args=ap.parse_args()
    status_path=f"{args.output}.status.json"
    status={'dataset':args.dataset,'output':args.output,'stage':'starting'}
    write_payload(status_path,status)
    gm.ENABLE_TRANSITION_RULES=False
    with h5py.File(args.dataset,'r') as f:
        status['stage']='reading_hdf5'
        write_payload(status_path,status)
        env_meta=json.loads(f['data'].attrs['env_args'])
        grp=f['data/demo_0']
        actions=np.asarray(grp['actions'])
        states=np.asarray(grp['states'])
        target_objects=args.target_object
        if target_objects is None and 'datagen_info/object_poses' in grp:
            target_objects=list(grp['datagen_info/object_poses'].keys())
        total=len(actions)
    end=total if args.end<0 else min(args.end,total)
    status.update({'stage':'creating_env','total_actions':int(total),'start':int(args.start),'end':int(end)})
    write_payload(status_path,status)
    env=RobomimicUtils.create_env(env_meta, init_curobo=args.init_curobo)
    status['stage']='env_created'
    write_payload(status_path,status)
    records=[]
    video_writer=None
    try:
        # Reset then restore exact generated initial state.
        if args.call_wrapper_reset:
            status['stage']='resetting'
            write_payload(status_path,status)
            env.reset()
        status['stage']='loading_initial_state'
        write_payload(status_path,status)
        og.sim.load_state(th.as_tensor(states[args.start]), serialized=True)
        if args.video_output:
            Path(args.video_output).parent.mkdir(parents=True, exist_ok=True)
            video_writer=imageio.get_writer(args.video_output, fps=args.video_fps)
            # Let the restored state propagate into the viewer camera before the first frame.
            for _ in range(3):
                og.sim.render()
            video_writer.append_data(_capture_frame(env, target_objects=target_objects))
        for i in range(args.start, end):
            status.update({'stage':'replaying','current_step':int(i),'num_records':len(records)})
            write_payload(status_path,status)
            if i == args.start or ((i-args.start) % max(1,args.snapshot_every)==0):
                records.append(_snapshot(env, i, ref_state=states[i], target_objects=target_objects))
            env.step(actions[i])
            if video_writer is not None and ((i - args.start + 1) % max(1,args.video_every)==0):
                video_writer.append_data(_capture_frame(env, target_objects=target_objects))
        records.append(_snapshot(env, end, ref_state=states[end] if end < len(states) else None, target_objects=target_objects))
        if video_writer is not None:
            video_writer.close()
            video_writer=None
        payload={'dataset':args.dataset,'start':args.start,'end':end,'video_output':args.video_output,'summary':summarize(records),'records':records}
        write_payload(args.output,payload)
        status.update({'stage':'completed','num_records':len(records),'summary':payload['summary']})
        write_payload(status_path,status)
        print(json.dumps(payload['summary'],indent=2))
        print(args.output)
        sys.stdout.flush()
    except BaseException as e:
        payload={
            'dataset':args.dataset,
            'start':args.start,
            'end':end,
            'error':f'{type(e).__name__}: {e}',
            'traceback':traceback.format_exc(),
            'records':records,
            'summary':summarize(records),
        }
        write_payload(args.output,payload)
        status.update({'stage':'failed','error':payload['error'],'traceback':payload['traceback'],'num_records':len(records),'summary':payload['summary']})
        write_payload(status_path,status)
        print(json.dumps({'error':payload['error'],'summary':payload['summary']},indent=2), file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
        try:
            status['stage']='closing_env'
            write_payload(status_path,status)
            env.close()
        except Exception:
            pass
        try:
            status['stage']='shutdown'
            write_payload(status_path,status)
        except Exception:
            pass
        og.shutdown()

if __name__=='__main__':
    main()
