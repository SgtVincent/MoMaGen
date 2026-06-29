# AGENTS.md

This file gives agents and developers repo-local guidance for working in the
MoMaGen checkout at `/home/ubuntu/repo/MoMaGen`.

## Project Scope

This repository contains MoMaGen code for constrained data generation for mobile
manipulation, plus a vendored `robomimic` tree and local task/config work for
BEHAVIOR-1K integration.

Current local focus:

- MoMaGen source-demo preparation and data generation.
- OmniGibson bimanual interfaces.
- CuRobo motion planning and attached-object handling.
- BEHAVIOR-1K task adaptation, especially `r1_turning_on_radio`.

The sibling BEHAVIOR workspace has its own instructions at
`/home/ubuntu/repo/BEHAVIOR-1K/AGENTS.md`; read that file when work crosses
into BEHAVIOR-1K, openpi-comet, or il_lib.

## Repo-Local Knowledge

TRAE CLI loads project-root `AGENTS.md` as project guidance. The CLI also has a
memory system controlled by `[memories]` / `/memories`, and shared resource
directories default to `.trae` and `.agents`.

For this repository, use the following convention:

- `.trae/documents/` is the repo-local long-term knowledge and document
  accumulation directory.
- `.trae/rules/` is for concise rules, review policies, and repeatable agent
  procedures.
- Root `AGENTS.md` is the index and policy entry point that tells future agents
  where to look.

The document below is an existing repo-local knowledge entry and should be read before
continuing BEHAVIOR-1K to MoMaGen bimanual contact work:

- `.trae/documents/b1k_to_momagen_bimanual_contact_tasks.md`

This location is appropriate for persistent repo-local knowledge because it is
inside the default TRAE resource directory and indexed here from `AGENTS.md`.
Without this `AGENTS.md` pointer, a standalone `.trae/documents` file may be
missed by future agents.

## How to Accumulate Knowledge

Add or update `.trae/documents/*.md` when a finding is reusable beyond the
current run, for example:

- task adaptation patterns,
- source-demo conversion requirements,
- config semantics,
- command recipes with caveats,
- known blockers and their evidence,
- validation gates and admission criteria,
- cross-repo integration notes.

Prefer one focused file per topic. Use a stable, searchable filename such as
`b1k_to_momagen_bimanual_contact_tasks.md`.

Each knowledge file should include:

- scope and date/context,
- authoritative local paths,
- exact config or command semantics,
- what was verified vs inferred,
- known missing assets or blockers,
- next recommended checks,
- evidence paths for important experiments.

Keep these documents decision-oriented. Do not paste long chronological logs
unless the sequence itself is the reusable lesson.

When a new experiment supersedes older guidance:

1. Update the existing topic file instead of creating a near-duplicate.
2. Mark stale conclusions explicitly.
3. Preserve only the evidence needed to understand why the decision changed.
4. If the change affects day-to-day behavior, also update this `AGENTS.md` or a
   file under `.trae/rules/`.

## Task Startup Checklist

Before working on MoMaGen + BEHAVIOR-1K tasks:

1. Read this `AGENTS.md`.
2. Check `.trae/documents/` for a topic file that matches the task.
3. Check `.trae/rules/` for repo-local agent procedures.
4. Verify live code/configs before relying on old experiment paths under `/tmp`.
5. If the user asks for a progress or handoff update, also update the relevant
   Feishu/Lark document through the Lark docs tooling.

## Current BEHAVIOR-1K Integration Notes

For `r1_turning_on_radio`:

- Do not claim success from shifted-marker or backside-button contact.
- Treat true visible-button overlap, `robot_can_toggle_steps > 0`, task state
  success, and video review as fail-closed admission gates.
- When rerunning any key experiment, set `write_video=True` where the launcher
  or config supports it, and report the third-view plus sensor/obs-layout video
  paths with the result.
- Seed 0 alone is not enough for stability claims.
- Prefer generic coordinated bimanual/contact-rich fixes over task-specific
  hard-coded offsets.
- Continue the whole-body/coordinated planning direction unless deliberately
  running an ablation.

For OmniGibson/Isaac Sim on this machine:

- Prefer `DISPLAY=:10.0` when a display is needed.
- Use explicit headless settings where supported.
- Do not silently switch to a different DISPLAY when diagnosing Vulkan or GPU
  foundation startup failures.

## Editing and Verification

- Keep edits scoped to MoMaGen, unless the task explicitly crosses into sibling
  repos.
- Use existing config and task patterns before adding new abstractions.
- For Python changes, run targeted syntax checks such as:

```bash
python3 -m py_compile momagen/datagen/waypoint.py
```

- For docs-only updates, run:

```bash
git diff --check
```

Limit validation to what is meaningful for the change and feasible in the
current environment.
