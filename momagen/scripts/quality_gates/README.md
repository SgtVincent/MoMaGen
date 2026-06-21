# MoMaGen quality gates

This folder contains replay and admission checks for generated demonstrations.

## turning_on_radio admission gate

For generated `turning_on_radio` candidates, admission is intentionally stricter
than standard MoMaGen success. A candidate must pass action replay and must also
provide observation evidence for semantic review:

- third-view replay video for the press/contact window,
- observation-layout replay video with left wrist over right wrist on the left
  and head camera on the right,
- per-camera visibility metrics for the radio object and switch/contact marker.

Example:

```bash
python momagen/scripts/quality_gates/build_action_replay_gate_manifest.py \
  --candidate A216 \
  --dataset momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/demo_src_r1_turning_on_radio_task_D0/demo.hdf5 \
  --smoke momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_replay_smoke_0_5.json \
  --long-window momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_replay_press_760_805_video_diag.json \
  --near-checkpoint momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_replay_press_780_805_checkpoint_diag.json \
  --obs-layout-long-window momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_replay_press_760_805_obs_layout_diag.json \
  --obs-visibility-long-window momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_replay_press_760_805_obs_visibility_diag.json \
  --long-start 760 \
  --output momagen/datasets/generated/turning_on_radio/A216_seed5_frac100_slowpress_repeatability/quality_gate/A216_action_replay_admission_gate_v1.json
```

Default observation thresholds are conservative admission checks for the
critical contact window from `first_can_toggle_step` through
`first_toggle_value_step`. Full-window visibility is still reported, but
full-window misses are warnings instead of hard admission blockers because
pre-contact approach can legitimately begin outside a tight head-camera view.

- head camera marker in-frame rate must be at least `0.8`,
- head camera radio mean pixel fraction must be at least `0.01`,
- at least one wrist camera must have marker in-frame rate at least `0.8`.

If action replay succeeds but critical-window observation thresholds fail, the
manifest should recommend
`keep_as_replay_gated_candidate_pending_observation_quality_review` instead of
admitting the candidate for training.

## Generated-data admission / conversion preflight

After a candidate passes the MoMaGen action/observation gate and human semantic
review, run a no-simulator/no-training preflight before creating any
BEHAVIOR/openpi-comet training-candidate manifest:

```bash
python momagen/scripts/quality_gates/build_generated_data_preflight_manifest.py \
  --candidate A201_seed4_predicate_hold_repeatability \
  --dataset momagen/datasets/generated/turning_on_radio/A201_seed4_predicate_hold_repeatability/demo_src_r1_turning_on_radio_task_D0/demo.hdf5 \
  --momagen-gate momagen/datasets/generated/turning_on_radio/A201_seed4_predicate_hold_repeatability/quality_gate/A201_action_replay_admission_gate_v1.json \
  --output momagen/datasets/generated/turning_on_radio/A201_seed4_predicate_hold_repeatability/quality_gate/A201_generated_data_admission_preflight_v1.json
```

The preflight checks that the generated HDF5 has finite canonical R1Pro 23D
actions/states, that the MoMaGen gate admits the seed, that review videos exist,
and that critical-window visibility and human semantic review are present. It
also reuses the current openpi-comet strict admission contract and fails closed
unless there is a strict `p0_simulator_verifier_admission` report.

For A201 this is expected to report
`observation_qualified_not_conversion_eligible`: the seed is a good next
admission candidate, but it is not yet a `b1k_generated_data_training_candidate`
and must not be converted to RFT parquet until strict simulator admission and an
explicit generated-data lineage mapping are available.
