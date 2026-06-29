# BEHAVIOR-1K to MoMaGen Bimanual Contact Tasks

This note captures the current working knowledge for adapting BEHAVIOR-1K
source demonstrations into MoMaGen-style coordinated bimanual data generation.
It is written from the `r1_clean_pan` reference task and the current
`r1_turning_on_radio` debugging path.

## Scope

- Reference task: `r1_clean_pan`, the official MoMaGen coordinated bimanual
  contact-rich task.
- Target task in progress: `r1_turning_on_radio`, converted from BEHAVIOR-1K
  demo data.
- Main question: what MoMaGen already supports, and what task-specific
  adaptation is still needed for a new bimanual contact-rich task.

## Clean-Pan Reference Configuration

Official base configs:

- `momagen/datasets/base_configs/r1_clean_pan.json`
- `momagen/datasets/base_configs/r1_clean_pan_mimicgen.json`
- `momagen/datasets/base_configs/r1_clean_pan_skillgen.json`

Generated experiment configs:

- `momagen/datasets/configs/demo_src_r1_clean_pan_task_D0.json`
- `momagen/datasets/configs/demo_src_r1_clean_pan_task_D1.json`
- `momagen/datasets/configs/demo_src_r1_clean_pan_task_D2.json`
- corresponding `mimicgen` and `skillgen` variants under the same directory.

Expected processed source dataset:

- `momagen/datasets/processed_source_demos/r1_clean_pan.hdf5`

This HDF5 is not present in the current local checkout, so clean-pan can be
audited through code/configs here, but not replayed or regenerated locally
without copying the source dataset in.

The baseline clean-pan phase structure in `r1_clean_pan.json` is:

| Phase | Type | Left arm | Right arm | Purpose |
|---|---|---|---|---|
| `phase_1` | `uncoordinated` | no object reference | `object_ref=frying_pan_602`, `MP_end_step=280`, term `470` | Right arm reaches/grasps pan. |
| `phase_2` | `uncoordinated` | `object_ref=scrub_brush_601`, `MP_end_step=880` | `attached_obj=frying_pan_602`, `MP_end_step=1000` | Left gets brush while right holds pan. |
| `phase_3` | `coordinated` | `object_ref=robot_r1`, `attached_obj=scrub_brush_601`, `MP_end_step=1100` | `object_ref=robot_r1`, `attached_obj=frying_pan_602`, `MP_end_step=1100` | Both arms execute coordinated scrub. |

Important implication: the official coordinated clean-pan phase is not
represented as "left hand targets the held pan surface while right hand stages
the pan". Instead, both arm trajectories in the final contact phase are replayed
relative to the robot/torso reference frame while each arm carries its attached
object. This is a narrower and more already-authored coordination pattern than
the current radio-button staging problem.

## Clean-Pan Task Registration and Environment Handling

Task interface registration lives in `momagen/env_interfaces/omnigibson.py`:

- `TASK_CONFIGS["r1_clean_pan"]` tracks:
  - `frying_pan_602`
  - `scrub_brush_601`
  - `robot_r1`
- It also maps robot-specific torso names:
  - R1: `torso_link4`
  - Tiago: `torso_lift_link`
- `MG_R1CleanPan` is the bimanual environment interface class.

The same file registers `r1_turning_on_radio`, but currently tracks only:

- `radio_89`
- `coffee_table_koagbh_0`

There is no clean-pan-like `robot_r1` tracked object in the checked-in
turning-on-radio task config. If a radio phase is represented using
robot-relative coordinated replay, this config needs to be extended or supplied
through `MOMAGEN_TASK_CONFIG_OVERRIDES`.

Clean-pan also has task-specific environment logic in
`robomimic/robomimic/envs/env_omnigibson.py`:

- D0/D1 randomization uses task-specific object motion ranges.
- `r1_clean_pan` has special handling for `mimicgen` and `skillgen` BDDL checks.
- D1 handling rotates `scrub_brush_601` so the handle orientation is usable.
- D2 handling adds distractors.

Conclusion: clean-pan working in MoMaGen is evidence that the codebase can
support bimanual contact-rich generation, not evidence that a new task works
without task registration, source-data phase annotation, and environment
adaptation.

## Data Generator Semantics

Core phase parsing and source-demo retargeting are in
`momagen/datagen/data_generator.py`.

Relevant behavior:

- `object_ref` selects the object or robot frame used to transform the source
  demonstration segment into the generated scene.
- `attached_obj` declares the object expected to be held by an arm for that
  phase.
- `MP_end_step` selects the pre-contact or motion-planning endpoint inside a
  subtask.
- `obtain_attached_object(...)` maps actually grasped task objects to
  `{left,right}_eef_link` entries for CuRobo attached-object planning, with
  scale `0.9`.
- After phases, the generator can check expected vs actual attached objects and
  fail when they mismatch.
- There is an explicit clean-pan-specific comment near the branch that handles
  `object_ref in ["robot_r1", torso_link_name]`: `TODO: this is a hacky for
  handling clean pan task. Improve this`.

This means clean-pan uses a special robot-reference route that should be treated
as a reference pattern, but not blindly copied. For new contact tasks, first
decide whether each contact phase is better represented as:

1. object-centric replay against the manipulated object,
2. robot/torso-centric coordinated replay, or
3. a staged two-object problem where the held object pose and active contact
   finger pose must be solved jointly.

`turning_on_radio` currently falls into category 3 for the press phase.

## CuRobo and Motion-Planning Flow

The primary waypoint/MP implementation is in `momagen/datagen/waypoint.py`.

Important call pattern:

- Calls are normalized by `_compute_trajectories_with_paths(env.cmg, ...)`.
- The main arm MP call passes:
  - `target_pos`
  - `target_quat`
  - `is_local=False`
  - `max_attempts=50`
  - `timeout=60.0`
  - `ik_fail_return=10`
  - `enable_finetune_trajopt=True`
  - `finetune_attempts=1`
  - `success_ratio=1.0 / batch_size`
  - `attached_obj=planning_attached_obj`
  - `attached_obj_scale=planning_attached_obj_scale`
  - `self_collision_check=effective_arm_mp_self_collision_check`
  - `emb_sel=emb_sel`
- Successful paths are converted with
  `env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)`.

Embodiment selection:

- Bimanual arm MP defaults to arm-only selection in many paths.
- The local radio debugging direction has been to use DEFAULT/whole-body where
  needed, controlled by:
  - `MOMAGEN_WHOLEBODY_ARM_MP=1`
  - `MOMAGEN_WHOLEBODY_ARM_MP_MIN_PHASE`
  - `MOMAGEN_WHOLEBODY_ARM_MP_MAX_PHASE`
  - `MOMAGEN_WHOLEBODY_ARM_MP_COVER_REPLAY`

For bimanual contact-rich tasks, avoid reducing the problem to "move base,
then arm-only press" unless deliberately doing an ablation. The preferred
direction for radio remains whole-body or coordinated planning with attached
objects considered.

## Current Turning-On-Radio Configuration

The current local processed source demo is:

- `momagen/datasets/processed_source_demos/r1_turning_on_radio_raw_episode_00000010.hdf5`

The current experimental config examined during this audit is:

- `/tmp/momagen_turning_on_radio_round60/r1_turning_on_radio_task_D0_phase2_mp1364_attached_transform_ref_phase2_A16_phase1_no_retract.json`

Its phase structure is:

| Phase | Type | Left arm | Right arm | Purpose |
|---|---|---|---|---|
| `phase_1` | `uncoordinated` | no object reference | `object_ref=radio_89`, `MP_end_step=1094`, term `1162` | Right reaches/grips radio. |
| `phase_2` | `coordinated` | `object_ref=radio_89`, `MP_end_step=1364`, term `1434` | `attached_obj=radio_89`, `MP_end_step=1364`, term `1434` | Left should press radio while right holds it. |
| `phase_3` | `uncoordinated` | no object reference | `object_ref=coffee_table_koagbh_0`, `attached_obj=radio_89`, `MP_end_step=1600`, term `1776` | Right places/continues carrying radio relative to table. |

The key difference from clean-pan is phase 2:

- Clean-pan final coordination: both arms replay relative to `robot_r1`, each
  with an attached object.
- Radio phase 2: the active left hand targets a contact feature on the same
  object that the right hand is holding.

That makes the radio phase a held-object staging problem. The planner must pick
or validate a feasible held radio pose and a feasible active-finger pre-contact
pose under attached-object, holder-arm, active-arm, reachability, and collision
constraints.

## Why Button Offset Tuning Was Not Enough

The current radio failures should not be framed as "MoMaGen cannot do
contact-rich bimanual tasks." They are task adaptation and execution-chain
failures.

Known failure modes from the current run series:

- Shifted marker or backside-press successes are invalid. They can report
  apparent success while the left hand presses the wrong side of the radio.
- Pulling the marker and overlap target back to the true button
  (`marker_local_offset=0,-0.15,0`) produced `success_rate=0.0` and
  `max_robot_can_toggle_steps=0`, exposing that earlier successes did not
  satisfy true visible-button contact.
- Some videos showed the right hand holding the radio at a far pose such that
  the button was outside left-hand reach. A feasible plan should bring the held
  object into a coordinated staging pose before the active press.
- Open-loop MP success is not enough. The held object and marker can drift
  during execution, so admission must verify runtime contact and task state.

Admission for radio-button data should remain fail-closed:

- true visible-button overlap,
- `robot_can_toggle_steps > 0`,
- `radio_on` / task state success,
- third-view and sensor video review for any new contact-mode change.
- key experiment reruns should set `write_video=True` where supported, and the
  result handoff should include third-view plus sensor / observation-layout
  video paths.

The generated-data preflight now enforces this at the admission-manifest layer:

- `momagen/scripts/quality_gates/replay_momagen_generated_demo.py` records true
  overlap center, primary-overlap hit, finger distance to overlap, and
  `robot_can_toggle_steps`.
- `momagen/scripts/quality_gates/build_action_replay_gate_manifest.py`
  classifies long-window / near-checkpoint replay and visibility evidence.
- `momagen/scripts/quality_gates/build_generated_data_preflight_manifest.py`
  now fails generated replay admission unless the action replay gate contains
  primary overlap, positive `robot_can_toggle_steps`, `ToggledOn=True`, and
  task success. This keeps A201 admitted while blocking shifted-marker or
  no-contact candidates.

## Retired Coordinated Staging Prototype

A241-A250 explored a generic coordinated contact staging prototype in
`waypoint.py`. That prototype has now been removed from the active code path
after review because it was plan-only / diagnostic and never produced
admission-valid generated data for `turning_on_radio`.

Historical diagnostic result:

- A249 showed the main staging path failed with `TrajOpt Fail`, while the
  diagnostic variant with attached object and no self-collision succeeded for
  all batch indices.
- A250 enabled the generic effective self-collision relaxation when an attached
  object exists. Plan-only staging found candidates for subgoals `0.33` and
  `1.0` with `success_idx=[0,1,2,3,4,5]` and
  `effective_self_collision_check=false`.
- A250 still had overall `success_rate=0.0` because the later execution chain
  failed after staging. Therefore this only identified one planning constraint
  issue, not a complete task solution.

Do not rerun the retired env flags as if they are current. The useful lesson is
that the next implementation should be generic and coordinated, but should be
built as an execution-validated chain with true-button admission rather than as
a plan-only staging diagnostic.

## BEHAVIOR-1K Demo to MoMaGen Source Dataset Requirements

Conversion entry point:

- `momagen/scripts/prepare_src_dataset.py`

The script:

- copies the raw input HDF5 into `momagen/datasets/processed_source_demos/`,
- optionally preprocesses OmniGibson datasets through robomimic helpers,
- creates an OmniGibson `DataPlaybackWrapper`,
- instantiates the MoMaGen env interface by name and interface type,
- replays selected episodes,
- records `datagen_info` per timestep,
- writes `datagen_info` into each episode group when
  `--generate_processed_hdf5` is passed.

The generated `datagen_info` contains the structures MoMaGen needs, including:

- `eef_pose`,
- `object_poses`,
- `subtask_term_signals`,
- `target_pose`,
- `gripper_action`.

For BEHAVIOR-1K tasks, conversion usually requires:

1. A raw HDF5 that can be replayed by the current OmniGibson/BEHAVIOR runtime.
2. A matching MoMaGen env interface class, for example
   `MG_R1TurningOnRadio`, with `env_interface_type=omnigibson_bimanual`.
3. Correct task tracked-object names in `TASK_CONFIGS`, or an explicit
   `MOMAGEN_TASK_CONFIG_OVERRIDES` JSON override.
4. A valid processed HDF5 under `momagen/datasets/processed_source_demos/`.
5. A phase annotation config with correct `object_ref`, `attached_obj`,
   `MP_end_step`, `subtask_term_step`, `type`, and `retract_type`.
6. Sanity playback or video review to confirm phase boundaries match the
   source behavior.
7. For generated data admission, task-specific success and contact diagnostics
   that reject visually wrong but numerically lucky trajectories.

Machine note: on this workspace, OmniGibson/Isaac Sim commands should be run
with `DISPLAY=:10.0` or explicit headless settings where supported.

## What Is Still Needed for Turning-On-Radio

To fully adapt `turning_on_radio` as a MoMaGen-style coordinated bimanual task:

1. Decide and document the final phase semantics.
   - If phase 2 remains held-object button press, model it as coordinated
     held-object staging plus active-finger contact.
   - If using clean-pan-style robot-relative replay, add the required
     `robot_r1`/torso reference config and verify the source demo supports that
     representation.

2. Replace the retired A250 staging prototype with a smaller generic execution
   chain.
   - The next evidence must execute the staged / pre-contact trajectory, then
     verify true-button contact and `radio_on` state.
   - Do not count a candidate as progress if it only finds a plan without
     runtime contact and task success.

3. Keep contact target validation tied to the real button.
   - Do not count shifted/backside marker hits.
   - Add or preserve diagnostics that measure true visible-button overlap and
     `robot_can_toggle_steps`.

4. Produce review videos for any new passing candidate.
   - Third-view video.
   - Sensor/robot-view video.
   - Include the episode/config path and seed in the handoff.

5. Cross-seed validate.
   - Seed 0 alone is not sufficient for this task.
   - At minimum, compare the same config across seeds after the first true
     contact success is found.

6. Obtain clean-pan source HDF5 if direct local comparison is needed.
   - Required path: `momagen/datasets/processed_source_demos/r1_clean_pan.hdf5`.
   - Without it, the repo can only support static/code audit of clean-pan.

## 2026-06-25 A273-A277 Coordinated Pre-Contact Staging Evidence

This update records the current evidence for a generic coordinated
bimanual/attached-payload execution path. It supersedes the immediate
"continue tuning final contact offset" direction, but does not relax the
admission gate. A277 was initially promising on endpoint/task metrics, but has
now been downgraded to diagnostic-only after human video review found severe
trajectory-quality regressions.

Implemented generic guards:

- Coordinated multi-EE planned-FK admission: a CuRobo result is not accepted
  just because the primary EE target is reached. For coordinated phases, all
  explicit target EEs must be within the configured positional tolerance.
- Coordinated multi-EE hard validation after execution: the actually executed
  left/right EE poses are compared against their phase targets before accepting
  the phase.
- Attached-payload pair collision validation: the active arm is checked against
  the held object's attached CuRobo collision spheres, with a configurable
  margin. For the current radio runs, `0.02m` matched the visually bad
  left-arm/radio near-collision window better than a zero-margin test.
- None-safe MP status handling: CuRobo can return successful candidates with
  `status=None`; logging and retry code must not assume `status.value` exists.

Evidence:

- A273 completed the whole pipeline and set `ToggledOn=True`, but review video
  showed a non-admissible left-arm / right-hand-held-radio collision around
  27-30s.
- A275/A276 showed the new gates fail closed instead of accepting bad plans.
  Some candidates reached only the main EE target while the held-hand target
  was off by about `0.29m` to `0.65m`; other candidates reached both EEs but
  violated the attached-payload clearance margin.
- A277 changed the phase-2 MP endpoint from final contact frame `1364` to
  pre-contact / staging frame `1320`, keeping the same generic validation
  stack. This produced a complete trajectory with `success_rate=100%`,
  `task_success=true`, `phases_completed=3`, and a review video.
- A277 phase-2 validation passed with both EEs near target:
  left EE actual-target error about `0.000218m`, right EE about `0.000175m`.
  The attached-payload pair check passed the `0.02m` margin with minimum
  distance about `0.029245m`.
- Human review rejected A277 anyway: the robot moved out of the camera frustum
  and showed large redundant arm / torso motion, making held-radio collision
  and backside-contact review impossible.
- Offline trajectory-quality analysis localized the regression. A277 phase-2
  MP itself was clean, but the after-MP replay / contact-closure segment was
  highly inefficient:
  - generated phase2 after-MP `275..893`: radio path `2.7655m`, net `0.3340m`,
    path/net `8.28`;
  - right EE path `2.4557m`, net `0.2919m`, path/net `8.41`;
  - left EE path `4.1413m`, net `1.1446m`, path/net `3.62`.
- The key root cause is a phase-gating error in the A277 launcher: contact
  prealign and post-MP press were restricted to execution phase `1`, so they
  ran after the initial radio grasp rather than in the coordinated phase `2`.
  This inserted a 450-step contact prealign plus 100-step press before the
  intended coordinated press stage and explains the large looped motion.

Interpretation:

- The current radio blocker is not that MoMaGen cannot solve bimanual
  contact-rich tasks. It is that using the source final-contact frame as the
  CuRobo MP target can push a collision-free planner into a contact/near-contact
  posture that should instead be handled by a local guarded contact controller.
- For held-object button pressing, prefer this generic staging formulation:
  first solve a collision-free coordinated pre-contact pose for both EEs and
  the attached payload, then use a small local guarded press to close contact.
  The guarded press must be phase-gated to the coordinated contact phase, not
  the preceding grasp / carry phase.
- Do not make the fix radio-specific. The reusable abstraction is
  coordinated multi-EE validation plus attached-payload clearance plus a
  pre-contact/staging MP endpoint for contact-rich closure, plus a
  trajectory-quality gate that rejects large object/EE path inefficiency even
  if endpoint and task state report success.

Admission caveat:

- A277 is not admitted training data and should not be treated as the current
  best candidate. It is a useful diagnostic showing that endpoint/task success
  can hide unacceptable trajectory quality.
- Logs show final `ToggledOn=True` and task success, but sampled
  `toggle_debug` entries still have `robot_can_toggle_steps=0`. This remains a
  semantic logging/admission issue, but trajectory quality is the immediate
  blocker.
- The next run should be A278: same pre-contact endpoint and hard-validation
  stack, but contact prealign / post-MP press gated to coordinated phase `2`,
  with video output and the trajectory-quality diagnostic rerun.

Evidence paths:

- `momagen/datasets/generated/turning_on_radio/A273_seed6_payload_pair_collision_video/`
- `momagen/datasets/generated/turning_on_radio/A276_seed6_payload_margin002_none_safe_video/`
- `momagen/datasets/generated/turning_on_radio/A277_seed6_phase2_mp1320_payload_margin002_video/`
- `momagen/datasets/generated/turning_on_radio/A277_seed6_phase2_mp1320_payload_margin002_video/demo_src_r1_turning_on_radio_task_D0/videos/0000.mp4`

## 2026-06-25 A278-A281 Trajectory-Quality Recheck

This update supersedes treating A277/A281 endpoint success as sufficient.
Human review and offline trajectory-quality diagnostics show that
`turning_on_radio` still needs an execution-trajectory admission gate before
training/conversion.

Key evidence:

- A278 moved contact target correction / prealign / post-MP press gates from
  phase 1 to coordinated phase 2. That fixed the phase-1 misfire, but phase-2
  navigation still displaced the held radio by about `0.562m`, so the
  navigation acceptance guard failed closed with `ref_obj_displaced`.
- A279 added held-object phase navigation suppression for coordinated phase 2.
  It reached phase 3 and the coordinated multi-EE hard validation passed with
  left/right EE errors around `0.00158m` / `0.00237m`, but contact closure still
  failed: post-MP press kept the active left finger about `0.733m` to `0.746m`
  from the button marker.
- A280 forced contact-prealign primary to `left_eef_link`. It failed task
  success, and the left finger remained far from the marker, around `0.64m`.
  Offline quality showed phase-2 left EE path `0.573m` for only `0.011m` net
  motion, path/net about `52.34`, indicating local redundant motion rather than
  useful contact approach.
- A281 added a default-off CuRobo candidate joint-path quality ranking and
  enabled it only for the experiment. It produced `success_rate=100%`,
  `task_success=true`, `phases_completed=3`, and a review video, but offline
  generated-trajectory quality became worse than A277:
  - A277 phase2 radio path/net `8.28`; A281 phase2 radio path/net `20.26`.
  - A277 phase2 right EE path/net `2.37`; A281 phase2 right EE path/net `4.0`.
  - A281 phase2 after-MP radio path `4.991m` for `0.246m` net displacement.

Interpretation:

- The main quality regression is not explained by the selected CuRobo MP
  candidate alone. A281 reduced the selected candidate's joint-path score, but
  the executed/generated object and EE paths were still poor. Therefore, MP
  candidate ranking is diagnostic-only and should not be treated as the
  solution.
- The actionable blocker is now execution-level path efficiency after MP,
  especially `phase*_after_mp` segments where replay/contact closure can move
  the held radio and inactive EE through large loops while endpoint/task state
  still reports success.
- A277 and A281 are both diagnostic-only. Admission must require visible button
  semantics, task success, acceptable generated trajectory quality, and
  reviewable camera framing.

Current next direction:

1. Add or enforce an execution-trajectory quality gate on generated HDF5:
   reject excessive radio path/net, EE path/net, post-MP object path, and
   camera-frustum loss before human review.
2. Localize the `phase2_after_mp` source of radio path blow-up: decide whether
   it comes from replay, contact prealign, post-MP press, or ref-object pose
   preservation.
3. Keep CuRobo candidate joint-path ranking as a default-off diagnostic only,
   unless future evidence shows it correlates with generated object/EE path
   quality.
4. Do not continue tuning only button offsets or primary-link overrides until
   the execution-trajectory quality gate can fail A277/A281 automatically.

Evidence paths:

- `momagen/datasets/generated/turning_on_radio/A278_seed6_phase2_contact_gate_video/`
- `momagen/datasets/generated/turning_on_radio/A279_seed6_skip_held_nav_phase2_video/`
- `momagen/datasets/generated/turning_on_radio/A280_seed6_prealign_left_primary_video/`
- `momagen/datasets/generated/turning_on_radio/A281_seed6_candidate_quality_video/`
- `momagen/datasets/generated/turning_on_radio/A281_seed6_candidate_quality_video/demo_src_r1_turning_on_radio_task_D0/videos/0000.mp4`
- `momagen/datasets/generated/turning_on_radio/A281_seed6_candidate_quality_video/trajectory_quality/trajectory_topdown.png`
- `momagen/datasets/generated/turning_on_radio/A281_seed6_candidate_quality_video/trajectory_quality/trajectory_timeseries.png`

## 2026-06-24 A254-A257 Marker Cleanup and Source Direction Evidence

This update supersedes the earlier shifted-marker interpretation for
`turning_on_radio`.

Key cleanup:

- Removed the risky local/runtime predicate shift from
  `BEHAVIOR-1K/OmniGibson/omnigibson/object_states/toggle.py`.
- Removed `runtime_state_corrections.ToggledOn.marker_local_offset` from the
  local radio `wxnicr` metadata.
- The source replay extractor now fails closed: a `button_target` is valid only
  when `first_can_toggle_step` exists. `first_toggle_value_step` alone may come
  from restored raw state and is not contact evidence.

Current evidence:

- A254, before cleanup, was invalid: `first_can_toggle_step=null`,
  `max_robot_can_toggle_steps=0`, and the extracted target came from
  `first_toggle_value_step`.
- A255, after cleanup, recovered the true source live predicate chain:
  `first_can_toggle_step=1360`, `first_toggle_value_step=1363`,
  `max_robot_can_toggle_steps=24`, left finger
  `left_gripper_finger_link1`.
- The A255 source target is marker-local
  `[0.044180317, -0.039027327, 0.011602766]`.
- The A255 source approach direction is marker-local unit
  `[0.179628047, 0.946952335, 0.266486473]`.

A256 and A257 ran with videos under repo-local generated data:

- A256 used the A255 target but kept the old fixed `+X` post-MP press/seek
  direction. It failed with `success_rate=0.0`; phase 1 press reached a best
  finger-marker distance around `0.055m`, then regressed to around `0.074m` and
  stopped with `finger_no_progress`.
- A257 changed only the post-MP contact seek / contact-aware press direction to
  the A255 source approach direction. The log confirmed the configured
  direction was applied, but prealign placed the left finger much farther from
  the marker, around `0.21m`, and press again stopped with
  `finger_no_progress`. Final status remained `success_rate=0.0` with later
  phase-2 MP failure.

Interpretation:

- The original `OMNIGIBSON_TOGGLEDON_OVERLAP_*` path was a risky local/debug
  workaround, not an admission-valid task solution.
- Source-derived press direction is necessary evidence but not sufficient by
  itself. The larger current failure is coordinated staging/prealign: before
  guarded press starts, the active finger must already be in a feasible
  button-side pre-contact shell.
- Do not spend more iterations tuning only the post-MP press direction. The next
  generic fix should validate or solve the held-object staging pose and active
  pre-contact finger pose jointly, then let local guarded press do only a small
  contact-rich closure.

Evidence paths:

- `momagen/datasets/generated/turning_on_radio/A254_source_button_target_extraction/quality_gate/A254_source_replay_button_target.json`
- `momagen/datasets/generated/turning_on_radio/A255_source_button_target_after_marker_cleanup/quality_gate/A255_source_replay_button_target.json`
- `momagen/datasets/generated/turning_on_radio/A256_seed6_a219_source_button_target_video/`
- `momagen/datasets/generated/turning_on_radio/A257_seed6_a219_source_approach_dir_video/`

## Recommended Next Implementation Steps

1. Treat A277 and A281 as diagnostic-only. Keep training/conversion closed.
2. Preserve `write_video=True` / video output under
   `momagen/datasets/generated/` for every key rerun.
3. Add an execution-trajectory quality gate that fails A277/A281 automatically:
   reject excessive radio path/net, EE path/net, large post-MP object path, or
   camera-frustum loss.
4. Localize the phase2/phase3 `after_mp` motion source before more offset or
   primary-link tuning. Candidate joint-path ranking alone is not enough.
5. If execution fails after staging/pre-contact, inspect the exact failing phase
   and decide whether the staged object pose is not applied, not held, or not
   preserved through replay.
6. Add compact diagnostics to record:
   - held object pose before and after staging,
   - active fingertip pose before contact,
   - true button pose,
   - distance/overlap at press,
   - attached-object mapping used by CuRobo,
   - effective self-collision setting,
   - selected embodiment.
7. Only after true-button contact and trajectory-quality gates pass, render
   third-view and sensor videos and ask for human visual review.
8. If no execution candidate survives, consider switching radio phase 2 to a
   robot/torso-relative coordinated replay representation and compare against
   the clean-pan phase-3 pattern.

## Useful Local Paths

- `momagen/datasets/base_configs/r1_clean_pan.json`
- `momagen/datasets/base_configs/r1_clean_pan_mimicgen.json`
- `momagen/datasets/base_configs/r1_clean_pan_skillgen.json`
- `momagen/datasets/configs/demo_src_r1_clean_pan_task_D0.json`
- `momagen/env_interfaces/omnigibson.py`
- `robomimic/robomimic/envs/env_omnigibson.py`
- `momagen/datagen/data_generator.py`
- `momagen/datagen/waypoint.py`
- `momagen/scripts/prepare_src_dataset.py`
- `momagen/datasets/processed_source_demos/r1_turning_on_radio_raw_episode_00000010.hdf5`

## 2026-06-26 Execution-Trajectory Quality Gate

User video review rejected A277/A281 because the robot, torso, and arms make
large redundant loops and leave the camera frustum. This means held-radio
collision/backside contact cannot be inspected, so endpoint success and
`success_rate=100%` are not admission evidence by themselves.

Implemented a no-simulator gate:

- Script:
  `momagen/scripts/quality_gates/build_generated_trajectory_quality_gate.py`
- Input:
  `trajectory_quality/trajectory_quality_metrics.json` exported from
  `momagen/scripts/debug/export_generated_trajectory_quality.py`
- Output:
  `quality_gate/<candidate>_trajectory_quality_gate_v1.json`
- README updated:
  `momagen/scripts/quality_gates/README.md`

The gate fails closed on missing critical tracks, excessive held-object /
EE path-to-net ratios, long post-MP held-object path, excessive post-MP base
path, and missing camera-framing review. This is intentionally conservative for
coordinated bimanual contact-rich tasks: generated trajectories must be
reviewable before semantic/contact admission.

Regression results:

- A277 gate:
  `momagen/datasets/generated/turning_on_radio/A277_seed6_phase2_mp1320_payload_margin002_video/quality_gate/A277_trajectory_quality_gate_v1.json`
  - `accepted=false`
  - `admission_recommendation=do_not_admit_trajectory_quality_failed`
  - main blockers:
    `phase2_held_object_path_net_too_high`,
    `phase2_after_mp_held_object_path_too_long`,
    `phase2_after_mp_held_object_path_net_too_high`,
    `phase2_after_mp_right_eef_path_net_too_high`,
    `phase2_after_mp_base_path_too_long`,
    `phase3_after_mp_held_object_path_net_too_high`,
    `camera_framing_human_review_missing`.
- A281 gate:
  `momagen/datasets/generated/turning_on_radio/A281_seed6_candidate_quality_video/quality_gate/A281_trajectory_quality_gate_v1.json`
  - `accepted=false`
  - `admission_recommendation=do_not_admit_trajectory_quality_failed`
  - main blockers:
    `phase2_held_object_path_net_too_high`,
    `phase2_after_mp_held_object_path_too_long`,
    `phase2_after_mp_held_object_path_net_too_high`,
    `phase2_after_mp_right_eef_path_net_too_high`,
    `phase2_after_mp_base_path_too_long`,
    `camera_framing_human_review_missing`.

Concrete failing values:

- A277:
  - phase2 radio path/net `8.2798 > 5.0`
  - phase2_after_mp radio path `2.7655m > 1.25m`
  - phase2_after_mp radio path/net `8.2798 > 6.0`
  - phase2_after_mp right EE path/net `8.4115 > 8.0`
  - phase2_after_mp base_xy path `1.5601m > 1.25m`
  - phase3_after_mp radio path/net `73.8066 > 10.0`
- A281:
  - phase2 radio path/net `20.2595 > 5.0`
  - phase2_after_mp radio path `4.9905m > 1.25m`
  - phase2_after_mp radio path/net `20.2595 > 6.0`
  - phase2_after_mp right EE path/net `18.6195 > 8.0`
  - phase2_after_mp base_xy path `1.8819m > 1.25m`

Interpretation:

- The current blocker is not whether A277/A281 can finish the task predicate.
  They finish but are not visually or physically reviewable enough for
  admission.
- The failure localizes strongly to `phase2_after_mp`: held radio, right EE,
  and base move too much after the MP endpoint.
- The next implementation step should instrument or constrain the
  `phase2_after_mp` execution path: replay, contact prealign, post-MP press,
  and held-object/ref-pose preservation. Do not continue button-offset tuning
  until this gate can pass.
