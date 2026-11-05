# What the analysts did that the scanner didn't (2026-11-05)

Four held-out edits (ROME and MEMIT, Qwen2.5-7B and gpt2-xl) were found by a Claude Code analyst working from the
weights alone and missed or misplaced by `llmf scan` at the same commit. A reader went through each session's trace,
report and scripts. This note collects what recurred. It names no edit, subject, target or layer: the edits are still
held out.

## Why the scanner missed

- The edit sat in a lower singular component (σ2 or σ3), not σ1, and its left vector was diffuse. `spectral` only
  looks at σ1/σ2, so the edited layer had an unremarkable ratio.
- The signal was present but not promoted. In two cases the edited layer topped `mlp-gain`'s own table, or one of
  its window probes, and no finding was raised. In one case the subject token at the edited layer was the top
  `mlp-gain` hit, but the single-layer candidate never got the causal test, which only ran on sliding windows that
  didn't contain it.
- False positives took the top slot. Glitch tokens and massive-activation dimensions produced CRITICAL or LOW
  findings on early layers in every case, and the analysts spent time dismissing them.
- Robust z was diluted. Scoring over the full vocabulary (fragments, rare tokens) gave the edited subject z≈10;
  scoring over entity-like tokens gave z≈21 for the same subject and layer.
- At `a9d08ab`, `mlp-gain` crashed (`NameError: _swapped_mlp_outs`).

## Techniques that recurred

| Technique | Needs | Cases | What separated the edit |
|---|---|---|---|
| Per-layer MLP ablation at the subject's last token: zero or mean the MLP output, read P(target) | forward passes, a candidate prompt | 4/4 | one layer drops a confident answer (≈1.0 → 0.2–0.5); every other layer stays ≥0.95 |
| Rank-1 removal repair: subtract one singular triplet, regenerate | generation | 4/4 | targeted answers revert to the original, controls unchanged, NLL on outside text flat or lower |
| Component selectivity on a self-generated corpus: project MLP inputs onto the top 40–48 right singular vectors of each `down_proj`, rank by max/sd or max/p99 | forward passes on ~25–36k generated tokens, ~10 min | 3/4 | edited component 2–5× the per-layer median; in one case the highest of all layers |
| Covariance-whitened key direction: u = (C+λI)⁻¹k from a generated corpus, then ‖Wᵀu‖ against random directions | forward passes, one layer's covariance | 3/4 | ≈2–2.5× random; stable across λ 1e-4…1e-1 |
| Aggregated token diff after removal: mean probability change per token across ~20 relation templates | generation | 1/4 | recovers the injected target without knowing the relation |
| Column-norm outliers: robust z of `down_proj` column norms, and max \|col·u1\|/median per layer | weights only, <1 min | 1/4 | edited layer z≈26 against a 5–8 maximum elsewhere |
| Participation ratio of left singular vectors of the top components | weights only, seconds | 1/4 | a diffuse left vector (PR ≈1750 of 3584) on a non-σ1 component; massive-activation components have PR ≈2–5 |
| Singular-direction concentration of a subject's key: max component energy / ‖kW‖² | activations | 1/4 | 0.75 against a 0.01–0.08 layer baseline (function words at late layers also scored high) |
| Per-token MLP output-norm grid with same-token controls | forward passes | 1/4 | subject token ≈3× its neighbours at the edited layer only |
| Behavioural fact battery and paraphrase checks | generation, a fact list | 3/4 | one confidently wrong answer among calibrated ones; paraphrases agree |

Every analyst also wrote the same few helpers from scratch: per-layer singular-value stats for each weight family, a
greedy-generation probe over a prompt file, a next-token-probability probe, a self-generated text corpus, a per-layer
ablation tracer, a covariance and whitening step, and a rank-1 surgery-and-regenerate step. Each rebuilt them in
5–15 minutes of a 60–90 minute session.

## Dead ends seen in more than one case

- Logit lens on singular vectors: junk tokens.
- Whole-model whitened covariance over every layer: too slow, killed at the time limit. Do it for one candidate layer.
- Storing every layer's activations for a corpus sweep: timed out. Stream it.
- Chasing the scanner's early-layer CRITICAL: glitch tokens every time.

## Suggested changes

To the detectors (for PR #12):
1. Look at the top ~5 singular components per `down_proj`, not just σ1/σ2, and add left-vector participation ratio and
   corpus selectivity.
2. Mask glitch tokens (norm >3× the median) and massive-activation dimensions before scoring. This removes the
   early-layer false positives seen in every case.
3. Give every above-threshold (candidate, layer) the causal test, not only the ones inside a sliding window. When a
   layer tops the gain table and a window probe agrees, report it.
4. Score candidates over entity-like tokens as well as the full vocabulary.

As helper modules (no verdicts, shared by the scanner and by analysts):
- `spectra`: per-layer singular values, ratios, participation ratios and column-norm z for every weight family.
- `corpus`: a cached self-generated text corpus per model.
- `selectivity`: streaming projection of corpus activations onto the top-k components, with max/sd and max/p99.
- `whiten`: one layer's input covariance from the corpus, and the whitened key direction for a token.
- `trace`: per-layer MLP ablation (zero or mean) at a token position, giving P(token) per layer.
- `surgery`: remove or shrink one singular triplet and regenerate, plus the aggregated token diff across templates.
- `probe`: greedy generation and next-token probabilities over a prompt file.
