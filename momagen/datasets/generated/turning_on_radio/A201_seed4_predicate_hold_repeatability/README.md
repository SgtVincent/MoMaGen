# A201 turning_on_radio candidate

This folder contains the first replay-gated MoMaGen candidate for the
BEHAVIOR-1K `turning_on_radio` task.

## Status

- Candidate: `A201_seed4_predicate_hold_repeatability`
- Standard generation result: `success_rate=100.0`, `num_success=1`,
  `ep_lengths=[811]`
- Replay gate verdict: `admit_candidate_after_human_review`
- Main caveat: the quality record shows the task predicate is already true at
  Phase 2 entry, followed by a Phase 2 TrajOpt failure. Treat this as a
  candidate requiring semantic review, not as an automatically admitted training
  sample.

## Replay gate evidence

The action replay admission gate checks three windows:

- `A201_replay_smoke_0_5.json`: short state-restore smoke check,
  max state error `0.003321979194879532`
- `A201_replay_press_780_811_overlap_diag.json`: long pre-contact / press
  window, reproduces the 5-step toggle hold and reaches `ToggledOn=True` at
  step `811`
- `A201_replay_press_805_811_overlap_diag.json`: near-contact checkpoint,
  also reproduces the 5-step hold and reaches `ToggledOn=True` at step `811`

The admission gate summary is in
`quality_gate/A201_action_replay_admission_gate_v1.json`.

## Contents

- `demo_src_r1_turning_on_radio_task_D0/demo.hdf5`: generated candidate demo
- `demo_src_r1_turning_on_radio_task_D0/mg_config.json`: generation config
- `demo_src_r1_turning_on_radio_task_D0/important_stats.json`: standard
  generation stats and phase logs
- `demo_src_r1_turning_on_radio_task_D0/logs/attempt_00001_succ_1_rate_100.0.json`:
  source attempt log for the successful rollout
- `quality_gate/*.json`: replay gate inputs and summary
