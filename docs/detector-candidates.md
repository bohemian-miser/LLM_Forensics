# From analyst discoveries to scanner detectors

A useful analysis should become a reusable candidate, not remain buried in a trace.
The existing `mlp-gain` detector is an example of a method harvested from an analyst trace.
A candidate can improve an existing detector rather than add a competing implementation.

## Agent deliverable

After completing the investigation report, if a method appears reusable, write
`detector_candidates/<id>/` inside the analyst output directory with these five files:

| File | Contents |
|---|---|
| `manifest.json` | Versioned metadata, method, provenance, applicability and limitations |
| `detector.py` | A `Detector` subclass using `ScanContext`, with a matching `id` and `default = False` |
| `test_detector.py` | Small synthetic positive and negative tests; no downloaded weights needed for unit tests |
| `README.md` | Algorithm, assumptions, how to reproduce, intended integration and existing detectors it improves |
| `validation.json` | Actual tests attempted, successes/failures and runtime/memory observations; explicitly mark untested work |

Manifest fields (schema version 1):

```json
{
  "schema_version": 1,
  "id": "method-name",
  "summary": "What this candidate detects",
  "method": "The algorithm and proposed threshold rule",
  "generalization": "Why it should work beyond the discovery suspect",
  "requires_base": false,
  "known_limitations": ["Unvalidated on other architectures"],
  "provenance": {
    "source_suspect": "THE_CURRENT_SUSPECT_ID",
    "reference": "Trace/tool steps or relative script paths supporting this method"
  }
}
```

Validation fields:

```json
{
  "status": "untested",
  "tests_run": [],
  "resources": {"device": null, "peak_vram_gib": null, "runtime_s": null}
}
```

Use `tested` only when `tests_run` records what was actually run and its outcome. Measurements
may remain null if unavailable. A tested claim is agent-reported, not collector certification.
Do not invent held-out results or weaken blinding to obtain control models.

Finish the investigation first; candidate preparation must not consume the entire run budget.
If nothing generalises, say so in the report rather than manufacturing a detector.
Do not hardcode the discovery suspect's subject, target, layer, ID or known answer into detection.
Keep provenance metadata separate from runtime decisions. Preserve failures and negative evidence.
Use token IDs as well as text; do not discard unfamiliar Unicode/byte tokens solely by appearance.
Use bounded batches, appropriate no-grad scopes, explicit memory budgeting and documented fallbacks.
Do not access answer records, install the candidate into the scanner or change shared tools.

The neutral instructions for the fleet owner to add to the analyst prompt are in
`lab/cleanroom/detector_candidate_contract.md`; copying them into a new run is owned by the fleet
coordinator. Existing active prompts are not changed by this feature.

## Collection and recurring review

The cleanroom runner preserves the whole analyst output and collates submissions after each
collection into `<workdir>/detector-review/`. The live fleet driver must call the same collector
at its own collection/batch boundaries. To collate again manually:

```bash
python -m lab.detector_candidates ~/llmf-batches/BATCH/traces \
  --out ~/llmf-batches/BATCH/detector-review
```

The collector reads only the five submission files, parses syntax/metadata, snapshots them by
content hash and writes `index.json` and `index.md`. It never imports candidates, runs their tests,
installs plugins, creates PRs or promotes detectors. Invalid submissions are reported without
blocking other candidates. Originals remain in the collected traces. Earlier snapshots remain
when an agent revises/removes a submission. An out directory must be outside the input traces.
There is no time-based scheduled service: collation is triggered by collection or an explicit rerun.

Review the accumulated index at batch completion and periodically across batches. Group related
ideas manually before promotion; matching file hashes alone do not establish algorithmic equivalence.
Do not expose cross-suspect candidate submissions to agents in an ongoing blind batch.

## Promotion gate

1. Review code and provenance, including any blinding contamination. Trace observations are discovery
   evidence, not an unbiased estimate of performance.
2. Run candidate tests in a controlled environment. Validate on unedited controls and held-out edits,
   using different subjects/layers; include multiple architectures when claiming portability.
3. Compare with existing detectors: false positives, sensitivity, localisation, recovered trigger/target,
   runtime and peak memory. Keep failure/skip rates and test populations in the report.
4. Select/calibrate thresholds on development data, then evaluate on a separate held-out set. Do not
   tune on the final test suspects or label a discovery-suspect reproduction as held-out validation.
5. Open a reviewed PR integrating the generic algorithm and detector adapter, with regression tests,
   benchmarks, attribution and explicit limitations. Reuse shared tools rather than duplicate them.
6. Keep `default = False` until evidence supports enabling it. Rerun scanner guard and compatibility
   tests before registration. Preserve original submissions and validation evidence after promotion.
