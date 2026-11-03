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

from llm_forensics.arch import mlp_out_param_name, mlp_out_weight_kv, num_layers
from llm_forensics.scan import Detector, DetectorResult, Finding, Severity, register_detector


@register_detector
class SigmaOutlier(Detector):
    id = "sigma-outlier"
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
| `ctx.suspect_spec`, `ctx.base_spec` | the weights paths as given on the command line |
| `ctx.tensors("suspect" \| "base")` | lazy tensor reader (no model instantiated) |
| `ctx.options.pick(fast, default, thorough)` | choose a setting by `--fast` / `--thorough` |
| `ctx.options.extra[id]` | per-detector overrides passed by callers of the Python API |
| `ctx.cache` | dict shared by all detectors in one scan (e.g. covariance per layer) |
| `ctx.log(msg)` | shown next to the detector's spinner |
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
The built-in detectors run their analysis in `llm_forensics/forensics/`, which returns a
`ForensicReport` dict (`forensics/report_schema.py`). The adapter in `scan/detectors/` turns it
into findings with `llm_forensics.scan.findings.from_report(report, self.id, prefix="SIGMA")` and
keeps the report itself in `DetectorResult.report`, which appears in the scan JSON. A new
detector can build findings directly, as above, or produce a report and use `from_report`,
which applies the shared severity rules:

| Report confidence | Severity |
|---|---|
| ≥ 0.9 | HIGH, or CRITICAL when both a trigger subject and an injected target were recovered |
| ≥ 0.6 | MEDIUM |
| ≥ 0.3 | LOW |
| below | INFO |

Changes outside MLP output projections are capped at MEDIUM. `llmf scan` exits 1 when any
finding is HIGH or CRITICAL.

`from_report` copies these evidence keys into the finding; the terminal report prints them
with the label shown:

| Key | Label | Meaning |
|---|---|---|
| `rank` | rank | numerical rank of the change |
| `delta_norm` | ‖ΔW‖ | Frobenius norm of the change (weight-diff) |
| `rel_norm` | ‖ΔW‖/‖W‖ | change relative to the base tensor (weight-diff) |
| `score` | layer score | best channel statistic divided by its threshold; ≥ 1 flags the layer (spectral) |
| `pcs_robust_z` | row-cosine z | pairwise cosine of MLP rows, robust z against the other layers (rank-reversal) |
| `r12` | σ1/σ2 | ratio of the top two singular values of W (spectral) |
| `key_odd` | key oddness | −log10 of how much ordinary text excites the top key direction, relative to an average direction (spectral) |

## LLM investigator toolsets

Each LLM investigator runs with one of two toolsets, chosen by `llmf scan --llm-mode`
(`tools`, `raw`, or `tools,raw` for both). The detector id and the report record which one ran:
`llm-anthropic` is the tools mode, `llm-anthropic-raw` the raw mode, and the report and
detector metrics carry `llm_mode: "tools" | "raw"`. The Python API takes
`investigate(..., llm_mode="raw")`.

`tools` (`investigator/tools.py`) gives the agent spectra, a logit lens, unembedding of
directions to their top tokens, and forward passes on prompts it writes.

`raw` (`investigator/raw_tools.py`) gives it only the stored numbers:

| Tool | Returns |
|---|---|
| `list_tensors`, `tensor_stats` | names, shapes; norms and top singular values per tensor |
| `diff_stats` (with `--base`) | suspect minus base per tensor |
| `read_block` | up to 16×16 raw values, with each column's mean and std over the whole tensor |
| `read_vector` | a row, column, singular vector or token row: largest entries (or chosen indices) next to the mean and std of the same position across its tensor |
| `top_entries` | the largest-magnitude entries of a tensor with their (row, col) |
| `norms` | per-row or per-column norms: quantiles, largest with robust z, smallest |
| `singular_vectors` | s1..s12 and their ratios, and the largest raw entries of the top singular vectors |
| `token_rows` | tokenization of texts and each token's embedding or unembedding row (dims that differ most from the mean row) |
| `compare_vectors` | dot and cosine between one vector and up to 32 others the agent names |

Numbers in one table share a power-of-ten scale printed in its header, and indices that run
over the vocabulary are labelled with their token. Outputs are capped at 4000 characters.

The rule that decides what belongs in raw mode:

1. Reading any stored tensor is allowed, and so is the tokenizer's id-to-string table.
2. Arithmetic on stored tensors is allowed at any size, provided it does not apply one tensor
   to another: slicing, sorting, norms, means, SVD of one matrix, suspect minus base.
3. Combining vectors is allowed only for vectors the agent names, at most 32 per call
   (`compare_vectors`). That is arithmetic it could do by hand, done for it.
4. Nothing that evaluates the network: no forward pass or generation, no matrix-vector or
   matrix-matrix product of weights, no nonlinearity or norm layer applied to anything, and no
   ranking of the vocabulary against a computed vector (unembedding a direction to its top
   tokens is a one-layer logit lens).

So a link from a direction to tokens has to come from the agent: guessing candidate tokens and
comparing their rows, or reading which dims dominate a vector and then which tokens are large
on those dims (a column of the embedding matrix is a stored vector, so `read_vector` may list
its top entries). `tests/test_investigator.py` runs every raw tool with `forward` and
`generate` patched to raise and a global forward pre-hook that raises.

## Edit records

A run directory from `llmf edit` also holds `edit_manifest.json` and `edit_vectors.safetensors`,
which say what was edited. `llmf scan` runs every detector inside
`llm_forensics.core.guard.guard_edit_records()`, so opening either file raises PermissionError
and the detector shows as ERROR.

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
