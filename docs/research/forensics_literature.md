# Forensics literature: detecting, locating, tracing and reversing weight edits

Research notes for the LLM_Forensics repo, 2026-11-03. Primary focus is Youssef, Zhao, Seifert,
Schlötterer, **"Tracing and Reversing Edits in LLMs"**, arXiv 2505.20819v2 (ICLR 2026), followed by the
other work that is directly usable here. Inputs each method needs are tagged:

* **S** = the suspect (edited) model only
* **B** = also needs the base / original weights
* **P** = needs the edit prompts, subjects, or a candidate pool containing them
* **A** = needs to run the model (activations), not just read the weights
* **T** = needs a labelled training set of edited models (supervised)

See `editing_methods.md` for the math of what the editors do to the weights, and `forensics_design.md`
for what this repo should build.

---

## 1. arXiv 2505.20819 — "Tracing and Reversing Edits in LLMs" (Youssef et al., Marburg / Sheffield)

### 1.1 What it does, in one paragraph

Threat model (Sec. 1–2): an attacker uses a knowledge-editing (KE) method to implant false facts, bias or
backdoors in a public model; the defender gets **only the edited model's weights**: no original weights,
no edit prompts, no knowledge of the editing operation. The paper defines two tasks on the edited MLP
projection matrix `W'_V` (= our `W_out` at the edited layer; the paper writes `W'_V = W_V + W_N`, `W_N` =
the update): **tracing** — recover the edited object `o'` — and **reversing** — restore the original
output `o` without knowing it. Both are evaluated mainly on single ROME / r-ROME edits.

### 1.2 Does it need the original weights? — the precise answer

**No, not for any of its three procedures, with one supervised caveat.**

| procedure | section | needs original `W_V`? | needs edit prompt? | other requirements |
|---|---|---|---|---|
| EditScope (tracing the object) | 4 | no | no | **T**: a training set of (edited matrix, edited object) pairs produced on the *same model with the same editing method* (600 train / 100 val, from 20 CounterFact relations), i.e. the defender must be able to *run the editor on an unedited copy* to build training data, even though that copy is never diffed against the suspect. |
| bottom-rank reversal | 5.1–5.3 | no | no | only the SVD of `W'_V`; but the cut-off `k` was chosen using ground-truth original outputs (Tab. 2) and the paper gives no unsupervised rule for `k`. |
| edited-layer localisation via row cosine (pcs) | App. C.2 | no (compares layers of the suspect model to each other / to typical values) | no | nothing else; fails on Llama-3. |
| relation prediction | App. D | no | no | **T**: logistic regression on a low-dimensional representation of edited matrices. |
| "unique predictions" edit detection | 5.3 | no | no | 100 wikitext-103 snippets; counts distinct 5-token greedy continuations as `k` varies. |

So the headline is right: the method operates on the suspect weights alone. The caveat is that
EditScope is supervised and model-specific, which in our setting is equivalent to having the base model
(we need it to generate training edits), so for the benchmark we should file EditScope under "needs base,
does not use it as a diff reference".

### 1.3 Algorithms

**EditScope (Sec. 4.1–4.2).** Keep the edited matrix `W'_Vi` frozen inside the edited model; add `m = 5`
new, randomly initialised token embeddings forming a fixed input `x_fixed`; train **all other parameters
of one layer at a time** (the edited layer or any later one — Fig. 16 shows accuracy is high when the
edited layer or an earlier layer is trained, and the paper trains "one layer at a time") with cross-entropy
so that `M(x_fixed)` outputs the edited object `o'_i`; AdamW lr 2e-5, β = (0.9, 0.98), wd 0.01, ≤ 100
epochs, early stopping patience 3. At test time the trained reader is applied to a new edited matrix and
the argmax token is the traced object. Interpretation: the rank-one update is so dominant that a probe
can learn to decode the planted object from the matrix with a fixed, content-free input.

**Rank-one analysis (Sec. 5.2, Eq. 1).** `M̃^{(k)} = Σ_i 1[i=k] Σ_ii u_i v_i^T` is the k-th rank-one
component of `W'_V = UΣV^T`. The maximum cosine between the rows of the update `W_N` and the rows of
`M̃^{(k)}` (10 relations) is **0.98 for GPT-2 XL at k=1**, 0.77 (k=1) / 0.45 (k=2) for GPT-J, and **0.20
for Llama-3** at k=1, rising over k = 2..4 then falling: the edit *is* the top singular component in
GPT-2 XL, mostly is in GPT-J, and is spread over the top few components in Llama-3.

**Bottom-rank reversal (Sec. 5.1–5.3, Eq. 2).** `M̃^{(r,k)} = Σ_{i>k} M̃^{(i)}`: drop the top `k`
singular components of the *edited* matrix and put the result back into the model. Measure editing
accuracy (should fall) and reversal accuracy (next-token prediction returns to the original object) and
KL to the original distribution. Reversal accuracy, ROME (Tab. 2):

| model (edited matrix shape) | k=0 | k=1 | best |
|---|---|---|---|
| GPT-2 XL (6400x1600) | ~0 | 87.1% | 94.5% at k=11 (editing acc. 1.3%) |
| GPT-J-6B (16384x4096) | ~0 | 32.3% | 80.0% at k=14 |
| Llama-3-8B (14336x4096) | ~0 | 5.5% | 80.0% at k=15 (editing acc. 6.5%) |
| Qwen2.5-7B (18944x3584) | ~0 | 31.9% | 62.9% at k=10/12/13 |

KL(original || reversed), Tab. 3: GPT-2 XL 6.04 → 0.009, GPT-J 11.57 → 0.22, Llama-3 10.07 → 0.60,
Qwen 8.99 → 1.53 (k=0 → k=15). GLUE capability of the k=15 Llama-3 model is on par with the edited model
(Fig. 5). Recovered outputs are "semantically close" rather than identical (Tab. 4). Note the cost: dropping
the top 15 singular directions of a 4096-rank matrix is a crude operation that also removes genuine
knowledge; it works because the edit lives in the top of the spectrum.

**Edit detection by "unique predictions" (Sec. 5.3).** 100 wikitext-103 snippets (≥ 50 chars), greedy
5-token continuations, for `k = 0..15`; count distinct continuations across `k`. Edited matrices are less
stable under truncation: GPT-J 2.46 (edited) vs 1.37 (unedited), Llama-3 1.84 vs 1.36. No threshold or
ROC is given; treat it as an idea, not a detector.

**Edited-layer localisation (App. C).** C.1: for a rank-one `W_N = u v^T` all rows of `W_N` are multiples
of `v^T`; in GPT models > 80% of the rows share the same sign, in Llama-3 it is ~50/50 (Fig. 18). C.2,
Eq. 8: `pcs(W) = 1/(n²−n) Σ_{i≠j} cos(w_i, w_j)` over the rows of the matrix (matrix in `[d_mlp, d_model]`
orientation per Tab. 21). pcs jumps from 9.1e-5 (unedited GPT-2 XL) to 0.0055–0.018 after a ROME edit
("more than 175x" for relation P190), from 1.9e-4 to 0.0016–0.004 for GPT-J ("more than 25x"), and does
**not** move for Llama-3 (edited ~ −3e-7 … −5e-7). The paper says this "can be used to examine and identify
edited layers" by comparison with unedited matrices. The Llama failure follows from C.1: with balanced
signs the common component cancels in a *signed* mean. (A sign-invariant version does not cancel; see
`forensics_design.md` B.2.)

### 1.4 Models, methods, results beyond ROME

Main text: ROME and r-ROME on GPT-2 XL, GPT-J-6B, Meta-Llama-3-8B, Qwen2.5-7B, CounterFact (31 relations
x 100 successful edits) plus a new YAGO-based set (15 relations, App. H). Tracing accuracy (in-domain /
out-of-domain relations), Tab. 1: ROME GPT-2 XL 99.4 / 99.7, GPT-J 97.6 / 94.4, Llama-3 96.5 / 91.2,
Qwen2.5 91.2 / 87.5; r-ROME within a few points of that. Appendix E: MEND (GPT-2 XL 99.5, GPT-J 99.7),
MEMIT (Qwen2.5 91.2, Mistral-7B 66.2), **AlphaEdit 1.9 / 2.8 (fails)**. Reversal: MEMIT single edits need
much larger `k` (Mistral 74.8% at k≈54–56, Qwen ≤ 55%); AlphaEdit needs `k` in the hundreds to thousands
(GPT-2 XL 81% at k=475, Llama-3 64% at k=2162) — consistent with AlphaEdit's update living in the
null space of the key covariance rather than on top of the spectrum. Batch (1000 facts) and sequential
(1000 facts, batches of 100) MEMIT/AlphaEdit edits are reversible to 47–81% (App. F–G).

### 1.5 What we take from it

* Single ROME edits are, in weight space, a **large rank-one spike at the top of the spectrum** of one
  matrix in GPT-2 XL / GPT-J, and a moderate one spread over the top ≈ 4–15 components in Llama-3 / Qwen.
* The planted object is decodable from the edited matrix alone (EditScope) — our `unembed_vector` tool on
  the top left singular vector is the training-free version of that.
* Row-cosine localisation is the only published *suspect-only* layer locator; it needs a sign fix for
  SwiGLU models and a cross-layer baseline.
* The paper does **not** report a detection ROC, does not recover the subject or the prompt, and does not
  address bf16 quantisation or multiple simultaneous edits of different sizes. Those are open for us.

---

## 2. Other directly relevant work

### 2.1 Reverse-engineering locate-then-edit updates with the base model — **B + P**

**KSTER — "Reverse-Engineering Model Editing on Language Models"** (Sun, Luo, Wang, Chen, He; arXiv
2602.10134, ICML 2026). The closest thing to our "with original weights" track. Given pre- and post-edit weights,
the covariance `C` the editor used, and a candidate pool of subjects / prompts: Stage I observes (via the
Woodbury identity on the MEMIT closed form) that the key-side row space of `ΔW · C` lies in the span of the
edit keys `K_1`; take the SVD of `M = ΔW C`, keep the top `N = rank(ΔW)` right singular vectors, and score
each candidate subject by the projection of its generic-template key onto that subspace. Stage II ranks
candidate prompts by the next-token entropy drop from pre- to post-edit model. Subject recall for MEMIT on
CounterFact with N = 10 / 50 / 100 edits: GPT-J 0.98 / 0.96 / 0.95, Llama-3-8B-Instruct 1.00 / 0.99 / 0.99,
Qwen2.5-7B 1.00 / 0.93 / 0.94; prompt recovery top-1 ≈ 0.5, top-5 ≈ 0.8–0.9, top-20 ≈ 0.95. They also
propose a defence ("subspace camouflage": mix decoy keys into `K_1`) that pushes the true subject's rank
from ≈ 51 to ≈ 394 at modest efficacy cost. Needs a candidate pool; no base-free variant. (Numbers as read
from the HTML; verify when citing.)

**"Watch the Weights" (WeightWatch)** (Zhong, Raghunathan; arXiv 2508.00161) — **B + A**. SVD of
`ΔW = W_ft − W_base` on every attention `o_proj` and MLP `down_proj`; the top-20 singular vectors per matrix
are "behaviour vectors"; inputs whose activations project unusually onto them (relative to a calibration
range from ~50k unlabelled samples) are flagged. Flags 56–100% of triggered backdoor completions on first
sight at < 1.04% FPR. They note that SVD of raw weights *without* subtracting the base "gave inconsistent
results" — a useful negative result for our no-base track.

**Intruder dimensions** (Shuttleworth, Andreas, Torralba, Sharma, "LoRA vs Full Fine-tuning: An Illusion of
Equivalence", arXiv 2410.21228) — **B**. A top-k singular vector of the fine-tuned matrix whose maximum
cosine with *all* base singular vectors is below ε is an "intruder"; LoRA creates them, full fine-tuning
almost never does. A ROME edit is an intruder by construction; untested on editing.

**Activation Difference Lens** (Minder et al., "Narrow finetuning leaves clearly readable traces in
activation differences", arXiv 2510.13900) — **B + A**. Mean residual difference (ft − base) at layer
`L/2` over the first few token positions of 10k random web texts, read with logit lens / Patchscope. An
agent given it identifies the fine-tuning objective for ~91% of 33 model organisms; mixing in ≥ 0.1:1
pre-training data largely removes the trace. For a single ROME edit the random-text footprint is probably
tiny (the key direction rarely fires), but the *subject-conditional* version is exactly our key-search tool.

**Crosscoders / model diffing** (Lindsey et al., transformer-circuits 2024; Minder et al. 2504.02922; Mishra-
Sharma et al. 2025) — **B + A**, heavy. Shared dictionaries with per-model decoders separate base-only /
ft-only features; built for chat-tuning-scale differences, untested on single-fact edits. Out of scope
for a first pass.

### 2.2 Detecting edited facts from activations — **S + A + P**

**"Has this fact been edited?" (DEED)** (Youssef et al., arXiv 2405.02765, NAACL 2025). Given the edited
model and a *prompt*, classify whether the produced fact is edited: features = last-token hidden state of
the last block + top-1000 next-token probabilities, AdaBoost on 126 labelled examples. F1 on GPT-J: ROME
98.0 (zsRE) / 90.7 (CounterFact), MEMIT 88.7 / 77.9; GPT-2 XL ROME 96.6 / 92.7. Needs you to ask the
edited fact — complementary to weight forensics (use it to *confirm* a candidate prompt).

### 2.3 Suspect-only weight anomaly detection — **S**

* **LoRA backdoor spectral signatures** (Puertolas Merenciano et al., arXiv 2602.15195). For each adapted
  module, five statistics of `ΔW = BA`: `σ_1`, `||ΔW||_F`, energy concentration `σ_1/Σσ_i`, spectral entropy,
  entry kurtosis → 20-dim signature; logistic regression on 400 benign / 100 poisoned adapters per backbone
  reaches ≈ 1.0 AUC on Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B. Norm statistics alone are weak (AUC 0.57–0.65);
  concentration / entropy / kurtosis carry the signal. Needs a labelled population; a template for a
  per-layer feature vector.
* **Weight Oracles** (Kabra, Venhoff, Schroeder de Witt, arXiv 2610.07334). An LLM fine-tuned to read raw
  weights and answer diagnostic questions; trained on benign pathologies, zero-shot AUROC 0.93 / 0.81 on
  backdoored small transformers; hand-crafted detectors "degrade under shift". Evidence that an
  LLM-in-the-loop investigator is viable at small scale.
* **MoTHer — model tree heritage recovery** (Horwitz et al., arXiv 2405.18432): weights-only lineage; the
  *kurtosis* of weight matrices decreases during fine-tuning, giving the parent → child direction; distance
  = mean layer L2 (full FT) or rank of ΔW (LoRA). Mostly vision models.
* **Fingerprinting** (HuRef 2312.04828: invariant products `X̂ W_Q W_K^T X̂^T`, `X̂ W_1 W_2 X̂^T` on rare-token
  embeddings; GhostSpec 2511.06390: normalised singular values of `W_Q W_K^T`, `W_V W_O`; REEF 2410.14273:
  representation CKA). Lesson: global similarity to the base is essentially 1.0 after fine-tuning (HuRef:
  Alpaca 99.91% cosine to Llama-7B), so a rank-one edit is invisible to global statistics — detection must be
  per-tensor and spectral.
* **Tooling**: `assay` (per-tensor mean/std/RMS/kurtosis/sparsity, flags layers > 5 MADs from the cross-layer
  median — its docs warn clean transformers trip this); WeightWatcher / HT-SR (power-law exponent α of
  each layer's spectrum). Neither has an edit-detection evaluation.
* **Staats et al. 2410.17770**: fine-tuning information also hides in the *smallest* singular values —
  the tail, not only the top, deviates from random-matrix predictions.

### 2.4 The threat side (what we may want to plant later)

* **Stealth edits** (Sutton et al., arXiv 2406.12670): replace the smallest-ℓ¹ row of `W_in` with a trigger
  detector and the matching column of `W_out` with a payload; argued undetectable *behaviourally*; no
  weight-space analysis. Signature: one anomalous row/column, not a dense rank-one delta.
* **BadEdit** (2403.13355), **Concept-ROT** (2412.13341), **MEGen** (2408.10722), **DualEdit** (2506.13285):
  backdoors via MEMIT/ROME-style closed forms; none evaluates weight-space detectability.
* **Position paper** "Editing LLMs poses serious safety risks" (Youssef et al., arXiv 2502.02958): the
  motivation for this repo.

### 2.5 Gaps (why this repo is not redundant)

1. Nobody reports a detection ROC for single ROME/MEMIT edits from the suspect weights alone; 2505.20819
   reports tracing / reversal accuracies and a qualitative pcs localiser.
2. Subject recovery exists only with the base weights plus a candidate pool (KSTER); no base-free subject
   recovery, and no vocabulary-level search.
3. No published handling of bf16-quantised deltas, of the EasyEdit `mom2_adjustment=false` variant, or of
   sign-balanced (SwiGLU) keys in localisation.
4. No LLM-agent evaluation of weight forensics on real LLMs (Weight Oracles is a fine-tuned small reader;
   the weight-space-programming blog, below, is the strongest evidence that a frontier model can do this
   interactively).

---

## 3. "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; blog, rikiyat.github.io, Nov 2026)

Claim: a frontier LLM can **read** and **write** network weights directly, around a natural-language
goal, with little or no execution. Evidence relevant to us:

* **Reading without execution works on small nets**: from raw weights as text (17k-param MLP policies,
  24k-param diffusion MLPs) the model names the game / the generated shape; given only the four *middle*
  blocks of 7.2M-param character transformers (no embeddings, no unembedding, Python allowed, no forward
  passes) it names the training language and finds every induction head with no false positives.
* **Reading fails when the signal is distributed**: honest vs sycophantic Gemma-3-270M fine-tunes could not
  be told apart from weights alone (0/4), but were separated 3/3 once the model could request attention
  patterns on texts it wrote itself — i.e. **a small, targeted activation tool rescues weight reading**.
* **Writing / editing an LLM with a tool budget**: with a Python kernel over Qwen3.5-0.8B's weights and
  10 / 25 / 50 forward passes, it located a "looping" direction written by the MLPs of layers 15 and 19,
  subtracted it from 64 output columns in each (0.016% of weights) and lifted GSM8K 13.8% → 67.1%. The
  negative Gemma-3 result shows it cannot add an absent ability.
* **Methodology** they used that we should copy: ground truth hidden from the agent; a budget of forward
  passes that is counted and logged; hypothesis → test loops; confidence reported; held-out evaluation.

What this suggests for the investigator design (our inferences, not the blog's claims):

1. Give the agent **summaries and tools over the weights**, not the tensors (a 58M-entry `down_proj` does
   not fit in context; per-layer statistics and SVD summaries do).
2. **Separate "weights-only" from "may run the model"** as explicit modes, and count forward passes.
3. Provide a cheap way to turn a weight direction into text (`unembed_vector`, `logit_lens`) and a cheap way
   to turn text into a weight-space query (`search_tokens_by_key`, `run_prompt`); the blog's successes all
   came from being able to cross the weight ↔ behaviour boundary in both directions.
4. Expect the agent to pattern-match (outline-vs-filled-shape error); require it to **verify by
   generation** before it reports a subject/target.

Sources: arXiv 2505.20819 (v2 HTML), 2405.02765, 2602.10134, 2508.00161, 2410.21228, 2510.13900,
2602.15195, 2610.07334, 2405.18432, 2312.04828, 2511.06390, 2406.12670, 2403.13355, 2502.02958,
transformer-circuits.pub/2024/crosscoders, rikiyat.github.io/blog/weight-space-programming/.
