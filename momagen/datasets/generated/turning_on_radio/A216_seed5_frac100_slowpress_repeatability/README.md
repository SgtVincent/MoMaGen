# A216 turning_on_radio candidate

This folder contains a replay-gated MoMaGen candidate for the BEHAVIOR-1K
`turning_on_radio` task.

## Status

- Candidate: `A216_seed5_frac100_slowpress_repeatability`
- Standard generation result: `success_rate=100.0`, `num_success=1`,
  `ep_lengths=[805]`
- Replay gate verdict:
  `keep_as_replay_gated_candidate_pending_observation_quality_review`
- Human semantic review: pending. The third-view replay video covers the
  `760..805` press / contact window and reproduces the switch toggle. The
  observation-layout replay video shows the policy-observation cameras for the
  same window: left wrist over right wrist on the left, head camera on the
  right.
- Main caveat: the source HDF5 episode is marked `partial=True`, and the
  initial replay snapshot reports task success before the physical toggle
  value changes. Treat this as a replay-gated candidate requiring semantic
  review, not as an automatically admitted training sample.
- Observation-quality caveat: the head camera sees the radio in every replayed
  frame, but the switch/contact marker is in the head-camera frame for only
  1/46 frames in the `760..805` press window. This candidate is useful as a
  replay-gated success case, but should not be admitted for training until the
  camera framing is accepted by human review or improved in a follow-up run.

## Replay gate evidence

The action replay admission gate checks three windows:

- `A216_replay_smoke_0_5.json`: short state-restore smoke check,
  max state error `0.00703144446015358`
- `A216_replay_press_760_805_video_diag.json`: long pre-contact / press
  window, reproduces the 5-step toggle hold with
  `first_can_toggle_step=780`, `first_toggle_value_step=805`, and
  `max_robot_can_toggle_steps=5`
- `A216_replay_press_780_805_checkpoint_diag.json`: near-contact checkpoint,
  also reproduces the 5-step hold with `first_can_toggle_step=780`,
  `first_toggle_value_step=805`, and `max_robot_can_toggle_steps=5`
- `A216_replay_press_760_805.mp4`: 1280x720 H.264 replay video, 46 frames at
  12 FPS
- `A216_replay_press_760_805_obs_layout_diag.json`: same `760..805` window
  replayed while rendering the input-observation camera layout; summary matches
  the long-window replay with `first_can_toggle_step=780`,
  `first_toggle_value_step=805`, and `max_robot_can_toggle_steps=5`
- `A216_replay_press_760_805_obs_layout.mp4`: 672x448 H.264 observation-layout
  replay video, 46 frames at 12 FPS
- `A216_replay_press_760_805_obs_visibility_diag.json`: per-camera observation
  visibility gate for the same `760..805` window. Radio visibility / switch
  marker in-frame rates are: left wrist `38/46` and `29/46`, right wrist
  `45/46` and `45/46`, head `46/46` and `1/46`.

The admission gate summary is in
`quality_gate/A216_action_replay_admission_gate_v1.json`.

## Contents

- `demo_src_r1_turning_on_radio_task_D0/demo.hdf5`: generated candidate demo
- `demo_src_r1_turning_on_radio_task_D0/mg_config.json`: generation config
- `demo_src_r1_turning_on_radio_task_D0/important_stats.json`: standard
  generation stats and phase logs
- `demo_src_r1_turning_on_radio_task_D0/logs/attempt_00001_succ_1_rate_100.0.json`:
  source attempt log for the successful rollout
- `quality_gate/*.json`: replay gate inputs and summary
- `quality_gate/A216_replay_press_760_805.mp4`: replay video for semantic
  review
- `quality_gate/A216_replay_press_760_805_obs_layout.mp4`: observation-camera
  replay video for semantic review from the generated episode inputs
- `quality_gate/A216_replay_press_760_805_obs_visibility_diag.json`:
  observation-quality metrics for admission review
