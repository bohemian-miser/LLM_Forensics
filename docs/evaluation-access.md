# Recording original-weight access and paired evaluations

Original-weight access, scanner base comparison, and blinding are different facts.
The edited-model table displays these independently; absent metadata is not evidence of isolation.
A delta's base is required to reconstruct the suspect but need not be supplied to a detector
as an original/reference model. In a blind environment, reconstruct outside the sandbox and
expose only the resulting suspect checkpoint.

## Per-attempt access record

The coordinator, not the investigating agent, should write `agents/access.json` after the
preflight audit. Preserve it with the run's checksums and access-check evidence:

```json
{
  "schema_version": 1,
  "analyst": {"base_weights_access": "none", "verification": "verified"},
  "investigators": {"base_weights_access": "none", "verification": "verified"}
}
```

`base_weights_access` is `none`, `available`, or `unknown`. `verification` is `verified`
(there is an accompanying coordinator access audit) or `declared` (the intended condition,
not independently verified). Missing or unsupported metadata is shown as not recorded.
Include audit references, tool/model revisions and artifact hashes as extra provenance fields.
Never retroactively label access verified using a prompt declaration alone.

The scanner's `target.base` (fallback `options.base`) separately indicates whether a base
was supplied for comparison. An explicit null means no base was supplied; an absent field
means not recorded. A no-base report cannot prove that the agent had no original files
available through another path. Investigator filesystem access must be audited separately.

`agents/BLINDING.json` is a separate annotation. `contaminated: true` means repository exposure
was recorded; false means no such exposure was recorded by that audit, not proof of all forms
of isolation. Preserve historical traces and annotations. Do not rewrite old results as if
newly tightened conditions applied to them.

## Paired comparison

For each immutable suspect checkpoint, run two new independent attempts:

1. **Suspect only:** expose the suspect checkpoint, tokenizer/config, analysis runtime and
   scratch/output. No originals, coordinator model registry/cache, answer records, repository
   examples, other agent sessions or earlier findings. Enforce access and network restrictions.
2. **Original supplied:** same suspect and settings, plus an explicitly identified original
   checkpoint, tokenizer/config and neutral instructions naming its location. Pass `--base`
   for scanner comparison. No edit records or previous reports.

Use fresh sessions and separated caches/workspaces for the two arms. Keep tool/code/model versions,
analysis budgets and trial seeds comparable. Record original checkpoint revision and hashes;
verify that it is the real pre-edit model rather than an approximation or re-planting.
Reserve enough memory for the actual workload: two 7B checkpoints can exceed a 24 GiB GPU.
Use a suitable larger-memory worker or validated streamed/CPU base comparison, and record the
hardware, placement, runtime and memory overhead for both arms. Do not co-schedule them on an
occupied worker or silently skip detectors that do not fit.

Store outputs and access audits in distinct immutable attempt directories. Do not overwrite the
published historical result or pool scores from different weight generations. The current table
shows the selected published result and its access record; a multi-attempt paired-score view is
separate future work. Do not relabel one cell as both arms.

Evaluate detection, correct layer, trigger/target recovery, false positives on unedited controls,
failures/timeouts, cost and memory. Repeat trials where agent behaviour is stochastic. Label
which detectors require originals rather than counting their absence in the suspect-only arm
as a negative finding. All methods/thresholds must be fixed before final held-out evaluation.
