# A201 turning_on_radio candidate

This folder contains the first replay-gated MoMaGen candidate for the
BEHAVIOR-1K `turning_on_radio` task.

## Status

- Candidate: `A201_seed4_predicate_hold_repeatability`
- Standard generation result: `success_rate=100.0`, `num_success=1`,
  `ep_lengths=[811]`
- Replay gate verdict:
  `admit_observation_qualified_seed_for_generated_data_pipeline`
- Human semantic review: passed. The reviewer confirmed that both the action
  trajectory and observation-layout data are sufficiently clear for the
  `780..811` press / contact window.
- Generated replay admission: passed on the current asset version. The
  no-training preflight records `generated_replay_admitted=true` and now
  includes true-button replay admission as a hard generated-replay gate.
- Conversion / training status: still closed. The current preflight records
  `conversion_eligible=false` because this candidate has not been promoted
  through a BEHAVIOR strict simulator admission report for training conversion.
- Main caveat: the quality record shows the task predicate is already true at
  Phase 2 entry, followed by a Phase 2 TrajOpt failure. Keep this caveat in the
  admission record, but A201 is now the first observation-qualified seed for
  the generated-data pipeline.

## Replay gate evidence

The action replay admission gate checks three windows:

- `A201_replay_smoke_0_5.json`: short state-restore smoke check,
  max state error `0.003321979194879532`
- `A201_replay_press_780_811_overlap_diag.json`: long pre-contact / press
  window, reproduces the 5-step toggle hold and reaches `ToggledOn=True` at
  step `811`
- `A201_replay_fuller_699_811_headless_diag.json`: fuller Phase 1 tail /
  post-arm-MP window, reproduces the final toggle with
  `first_can_toggle_step=809`, `max_robot_can_toggle_steps=5`, and
  `ToggledOn=True` at step `811`
- `A201_replay_press_805_811_overlap_diag.json`: near-contact checkpoint,
  also reproduces the 5-step hold and reaches `ToggledOn=True` at step `811`
- `A201_replay_press_780_811.mp4`: 1280x720 H.264 third-view replay video for
  the press / contact window
- `A201_replay_press_780_811_obs_layout.mp4`: 672x448 H.264 observation-layout
  replay video for the same window
- `A201_replay_press_780_811_obs_visibility_diag.json`: per-camera observation
  visibility gate for the same window. Full-window head marker visibility is
  only `15/32`, so it remains a warning. The critical contact window
  `807..811` passes: head marker `5/5`, head radio mean pixel fraction
  `0.0322845458984375`, and right-wrist marker `5/5`.

The admission gate summary is in
`quality_gate/A201_action_replay_admission_gate_v1.json`.

`quality_gate/A201_generated_data_admission_preflight_v1.json` records the
generated-data admission / conversion preflight. It now separates the two
decisions explicitly:

- `generated_replay_admitted=true`: A201 is accepted as an
  observation-qualified generated replay seed on the current asset version.
- `true_button_replay_admission.accepted=true`: the long-window replay records
  primary overlap at step `807`, `max_robot_can_toggle_steps=5`,
  `ToggledOn=True` at step `811`, task success observed, and best left-finger
  distance to true overlap `0.01427403846702593`.
- `conversion_eligible=false`: A201 is not yet a training/conversion candidate.
  The remaining blocker is the missing promotion-time BEHAVIOR strict simulator
  admission report.

The fuller-window diagnostic was generated in headless mode because the local
Kit viewer path intermittently fails during swapchain initialization. This is a
video/display environment issue, not a replay failure: the headless replay
completed and reproduced the toggle.

## Contents

- `demo_src_r1_turning_on_radio_task_D0/demo.hdf5`: generated candidate demo
- `demo_src_r1_turning_on_radio_task_D0/mg_config.json`: generation config
- `demo_src_r1_turning_on_radio_task_D0/important_stats.json`: standard
  generation stats and phase logs
- `demo_src_r1_turning_on_radio_task_D0/logs/attempt_00001_succ_1_rate_100.0.json`:
  source attempt log for the successful rollout
- `quality_gate/*.json`: replay gate inputs and summary
- `quality_gate/A201_generated_data_admission_preflight_v1.json`: no-training
  admission/conversion preflight for the generated-data pipeline
- `quality_gate/A201_replay_press_780_811.mp4`: third-view replay video for
  semantic review
- `quality_gate/A201_replay_press_780_811_obs_layout.mp4`: observation-camera
  replay video for semantic review from generated episode inputs
