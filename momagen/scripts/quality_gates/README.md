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

When rerunning a key generation experiment, set `write_video=True` where the
launcher or config supports it. Include the third-view and sensor /
observation-layout video paths in the experiment handoff so the sim execution
trajectory can be reviewed before admitting the candidate.

## Generated trajectory quality gate

For coordinated bimanual contact-rich tasks, endpoint success and task success
are not sufficient admission signals. A candidate can press the predicate while
the held object, inactive EE, base, or torso make large redundant loops that
leave the camera frustum and make contact semantics impossible to review.

Before action/observation replay admission, export no-simulator trajectory
metrics from the generated HDF5 and run the generated trajectory quality gate:

```bash
python momagen/scripts/debug/export_generated_trajectory_quality.py \
  --dataset momagen/datasets/generated/turning_on_radio/<candidate>/demo_src_r1_turning_on_radio_task_D0/demo.hdf5 \
  --output-dir momagen/datasets/generated/turning_on_radio/<candidate>/trajectory_quality

python momagen/scripts/quality_gates/build_generated_trajectory_quality_gate.py \
  --candidate <candidate> \
  --metrics momagen/datasets/generated/turning_on_radio/<candidate>/trajectory_quality/trajectory_quality_metrics.json \
  --require-camera-framing-review \
  --output momagen/datasets/generated/turning_on_radio/<candidate>/quality_gate/<candidate>_trajectory_quality_gate_v1.json
```

This gate currently fails closed on excessive held-object / EE path-to-net
ratios, long post-MP held-object path, excessive post-MP base path, missing
critical tracks, and missing camera-framing review. It is intentionally
conservative for `turning_on_radio`: A277/A281 style samples with 100% pipeline
success but large `phase2_after_mp` radio/EE loops must remain diagnostic-only
until the execution trajectory is visually reviewable.

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
that critical-window visibility and human semantic review are present, and that
the action replay gate contains true-button contact evidence. For
`turning_on_radio`, generated replay admission fails closed unless the replay
summary has primary overlap, positive `robot_can_toggle_steps`, `ToggledOn=True`,
and task success. It also reuses the current openpi-comet strict admission
contract and fails closed for training conversion unless there is a strict
`p0_simulator_verifier_admission` report.

For A201 this is expected to report
`observation_qualified_not_conversion_eligible`: the seed is a good next
admission candidate, but it is not yet a `b1k_generated_data_training_candidate`
and must not be converted to RFT parquet until strict simulator admission is
available. Its generated-data lineage is already compatible with the current
openpi-comet source/action contract.
