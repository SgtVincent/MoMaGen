# A216 turning_on_radio candidate

This folder contains a replay-gated MoMaGen candidate for the BEHAVIOR-1K
`turning_on_radio` task.

## Status

- Candidate: `A216_seed5_frac100_slowpress_repeatability`
- Standard generation result: `success_rate=100.0`, `num_success=1`,
  `ep_lengths=[805]`
- Replay gate verdict: `admit_candidate_after_human_review`
- Human semantic review: pending. The replay video covers the `760..805`
  press / contact window and reproduces the switch toggle.
- Main caveat: the source HDF5 episode is marked `partial=True`, and the
  initial replay snapshot reports task success before the physical toggle
  value changes. Treat this as a replay-gated candidate requiring semantic
  review, not as an automatically admitted training sample.

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
