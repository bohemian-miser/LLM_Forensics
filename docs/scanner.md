# Writing a detector for `llmf scan`

A detector is a class that receives a `ScanContext` and returns a `DetectorResult`.
The scanner handles loading weights, progress display, error handling, the report and
the JSON output; the detector only has to look at the weights and say what it found.

## A minimal detector

This one flags any MLP output projection whose largest singular value is far above the
median across layers.

```python
# my_pkg/norm_outlier.py
import statistics

from llm_forensics.model_io import mlp_out_param_name, mlp_out_weight_kv, num_layers
from llm_forensics.scan import Detector, DetectorResult, Finding, Severity, register_detector


@register_detector
class SigmaOutlier(Detector):
    name = "sigma-outlier"
    title = "Top singular value outlier"
    description = "Flags MLP output projections whose sigma_1 is far above the median layer."
    edit_types = frozenset({"rome", "memit"})
    cost = "low"                       # low | medium | high | api

    def run(self, ctx) -> DetectorResult:
        model = ctx.suspect.model      # loaded once per scan, shared with other detectors
        s1 = {l: float(mlp_out_weight_kv(model, l).float().svd().S[0]) for l in range(num_layers(model))}
        med = statistics.median(s1.values())
        ratio = {l: v / med for l, v in s1.items()}
        findings = [
            Finding(id=f"LLMF-SIGMA-{l:03d}", severity=Severity.MEDIUM, detector=self.id,
                    title="Singular value outlier", edit_type="rome", layer=l,
                    tensor=mlp_out_param_name(model, l), confidence=min(1.0, r / 10),
                    evidence={"sigma1": s1[l], "ratio_to_median": r})
            for l, r in ratio.items() if r > 3
        ]
        top = max(ratio, key=ratio.get)
        return DetectorResult(findings=findings, detected=bool(findings), layer=top if findings else None,
                              layer_scores={l: min(1.0, r / 3) for l, r in ratio.items()})
```

`layer_scores` (0 to 1 per layer) feeds the layer-anomaly strip; `detected` and `layer`
feed the agreement matrix. Everything except `findings` is optional.

## Registering it

Inside this repository, import the module from `llm_forensics/scan/detectors/__init__.py`.
From another package, declare an entry point; `llmf scan` loads the group on start-up:

```toml
[project.entry-points."llm_forensics.detectors"]
sigma-outlier = "my_pkg.norm_outlier:SigmaOutlier"
```

`llmf scan --list-detectors` shows it, and `llmf scan model/ --detectors sigma-outlier` runs it.
A plug-in that fails to import is listed as a warning at the end of the report.

## What a detector can use

| Attribute | Meaning |
|---|---|
| `ctx.suspect` | `.model`, `.tok`, `.name`, `.source_kind` of the weights under test |
| `ctx.base` | the same for `--base`, or `None` |
| `ctx.suspect_spec`, `ctx.base_spec` | the weight specs as given on the command line |
| `ctx.tensors("suspect" \| "base")` | lazy tensor reader (no model instantiated) |
| `ctx.options.pick(fast, default, thorough)` | choose a setting by `--fast` / `--thorough` |
| `ctx.options.extra[name]` | per-detector overrides passed by callers of the Python API |
| `ctx.cache` | dict shared by all detectors in one scan (e.g. covariance per layer) |
| `ctx.log(msg)` | shown next to the detector's spinner, kept in the scan log |
| `ctx.device` | resolved device |

Class attributes control when a detector runs:

- `requires_base = True`: selected by default only when `--base` is given; if named
  explicitly without `--base` it is reported as SKIPPED.
- `requires_env = ("MY_API_KEY",)`: skipped unless one of these variables is set.
- `default = False`: runs only when named in `--detectors`.
- `group`: an extra selector shared by several detectors (the LLM investigators use `llm`).
- `instances()`: override to contribute several detectors from one class; the LLM
  investigator returns one per provider with a key.

## Findings and severity

`Finding` fields: `id`, `severity`, `detector`, `title`, `edit_type`, `layer`, `tensor`,
`description`, `subject_candidates`, `target_candidates`, `confidence`, `evidence`.
Detectors that already produce a `ForensicReport` dict (`forensics/report_schema.py`) can
call `llm_forensics.scan.findings.from_report(report, self.id, prefix="SIGMA")`, which
applies the shared severity rules:

| Report confidence | Severity |
|---|---|
| ≥ 0.9 | HIGH, or CRITICAL when both a trigger subject and an injected target were recovered |
| ≥ 0.6 | MEDIUM |
| ≥ 0.3 | LOW |
| below | INFO |

Changes outside MLP output projections are capped at MEDIUM. `llmf scan` exits 1 when any
finding is HIGH or CRITICAL.

## Errors

Exceptions raised by `run()` are caught; the detector shows as ERROR with the message, and
the other detectors still run. Raise with a message a user can act on.

## Testing

```python
from llm_forensics.scan.context import ScanContext
from llm_forensics.scan.registry import run_detector

res = run_detector(SigmaOutlier(), ScanContext("gpt2"))
assert res.status == "OK"
```

`tests/test_scan.py` has examples that render a report to a string console.
