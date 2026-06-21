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

Default observation thresholds are conservative admission checks:

- head camera marker in-frame rate must be at least `0.8`,
- head camera radio mean pixel fraction must be at least `0.01`,
- at least one wrist camera must have marker in-frame rate at least `0.8`.

If action replay succeeds but observation thresholds fail, the manifest should
recommend `keep_as_replay_gated_candidate_pending_observation_quality_review`
instead of admitting the candidate for training.
