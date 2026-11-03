# Forensic design: how to detect, locate and explain a ROME/MEMIT-style edit

Design notes for `llm_forensics/forensics/` and `llm_forensics/investigator/`, 2026-11-03. Builds on
`editing_methods.md` (what the edit does to the weights) and `forensics_literature.md` (what is already
published). Pseudocode is PyTorch-flavoured and uses the helpers in `llm_forensics/arch.py`
(`mlp_out_weight_kv()` returns the edited matrix in math orientation `[d_model, d_mlp]`, `ArchSpec.final_norm`,
`ArchSpec.unembed`). Numbers marked [lit] come from the papers; numbers marked [unmeasured] are
expectations that have not yet been checked against this repo's edits.

Three tracks:

* **A. With the base weights** — a deterministic pipeline that should recover layer, rank, subject and
  target essentially exactly for a ROME edit, and per-edit for small MEMIT batches.
* **B. Weights only** — statistical anomaly detection across layers of the suspect model,
  plus the same key/target read-outs applied to the suspect's own top singular components.
* **C. LLM investigator** — a constrained tool set over A/B primitives, and an evaluation protocol in which the agent never sees what was planted.

---

## 0. What the edit looks like (recap) and what the forensics must output

For a single ROME edit at layer `l` the only change is

```
ΔW = W' - W = a b^T          a ∈ R^{d_model}  (write direction, added to the residual stream at the subject token)
                             b ∈ R^{d_mlp}    (read direction over the MLP hidden layer; ∝ C^{-1}k* or ∝ k*)
ΔW k(x) = a · ⟨b, k(x)⟩      the edit fires on any input whose layer-l key has a large inner product with b
```

MEMIT: the same at 5–6 consecutive layers, rank ≤ batch size each. See `editing_methods.md` Sec. 1.8, 2.3.

The forensic report we want, in decreasing order of difficulty:

1. **edited?** yes/no (and family: locate-then-edit vs fine-tune/LoRA vs clean)
2. **where**: tensor name(s) and layer(s); **rank** of the change per tensor
3. **what it reads**: the subject (string or token ids) and the prompt/relation that triggers it
4. **what it writes**: the target object (token(s))
5. **verification**: original vs edited outputs on the recovered prompt, and a specificity check

Keys at layer `l` are computed *before* `W_out^{(l)}` is applied, so **`k(x)` is identical in the original
and edited model** (for a ROME edit at a single layer). This is what makes the subject search clean: we
can use whichever model we have to compute keys.

---

## A. With the base weights

### A.1 Locate the changed tensors

```python
def diff_tensors(orig_path, edited_path):
    # stream safetensors shards; never load both full models into RAM at once
    rows = []
    for name in shared_tensor_names(orig_path, edited_path):
        W0 = load(orig_path, name).float(); W1 = load(edited_path, name).float()
        D = W1 - W0
        rows.append(dict(name=name, shape=tuple(D.shape),
                         frob=D.norm().item(), rel=(D.norm() / W0.norm()).item(),
                         nnz_frac=(D != 0).float().mean().item(), max_abs=D.abs().max().item()))
    return sorted(rows, key=lambda r: -r["frob"])
```

Expected outcome:

| planted change | tensors with `frob > 0` | what to print |
|---|---|---|
| ROME | exactly 1, and it is `ArchSpec.mlp_out` of one layer | name, `rel`, `nnz_frac` |
| MEMIT / AlphaEdit / PMET / BadEdit | 5–6 consecutive `mlp_out` tensors | names, per-layer `frob` profile |
| LoRA merged | q/k/v/o (± MLP) in every layer | count of changed tensors (≫ 10) |
| full fine-tune | everything, incl. LayerNorm and embeddings | — |
| clean | nothing (or only dtype-conversion noise) | — |

Report `rel = ||ΔW||_F / ||W||_F`. [lit] r-ROME Fig. 2–3 (GPT-J layer 5): normal ROME edits have
`||ΔW||_F` of order 5–20, disabling edits of order 10^3; for GPT-2 XL the update is larger than
`σ_1(W)` (2505.20819 Sec. 5.2). [unmeasured] `rel` for every model we use, for r-ROME vs original ROME.

`nnz_frac` tells you the dtype story: fp32-saved edits give `nnz_frac ≈ 1.0` on the edited tensor; a
bf16-saved edit gives a sparse delta (entries smaller than half the bf16 spacing at `W_ij`, i.e. below
0.2–0.4% of `|W_ij|`, round to zero). [unmeasured] `nnz_frac` for bf16 checkpoints; expect well below 1 for GPT-J/Llama-scale
matrices where `|ΔW_ij|` is tiny relative to `|W_ij|` for most entries.

### A.2 Rank and the two directions

```python
def analyse_delta(model_orig, model_edit, layer):
    W0 = mlp_out_weight_kv(model_orig, layer).float()      # [d_model, d_mlp]
    W1 = mlp_out_weight_kv(model_edit, layer).float()
    D  = W1 - W0
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)    # 1600x6400 takes < 1 s on GPU, ~10 s CPU; 4096x16384 ~ 1 min CPU
    energy = (S**2 / (S**2).sum())
    rank_est = int((S / S[0] > 1e-2).sum())                # components above 1% of the top one
    a = U[:, 0] * S[0]                                     # write direction (scaled)   R^{d_model}
    b = Vh[0]                                              # read direction  (unit)     R^{d_mlp}
    return dict(S=S[:16], energy_top1=energy[0].item(), rank_est=rank_est, a=a, b=b, U=U[:, :rank_est], V=Vh[:rank_est].T)
```

Expected: fp32 ROME → `S[1]/S[0] < 1e-4` (numerical zero), `energy_top1 > 0.9999`. bf16-saved ROME →
`S[1]/S[0]` in the 1e-3…1e-2 range: rounding adds a dense noise matrix whose Frobenius norm is roughly
`||W||_F · 2^-8 / sqrt(12) ≈ 1e-3 ||W||_F` and whose spectral norm is a further `~(1/sqrt(d_model) + 1/sqrt(d_mlp))`
smaller, so the rank-one spike still dominates by orders of magnitude [unmeasured]. MEMIT batch of `u`
→ `rank_est ≈ u` per layer (keys are nearly independent); for `u = 1` a rank-1 at each of 5–6 layers.

**Sign convention.** The SVD fixes only the product `a b^T`. Resolve by requiring `⟨b, k_subject⟩ > 0` for
the best subject key found in A.3 (for `mom2_adjustment=false` edits `b ∝ k*` so this is automatic); then
`+a` is the direction that *promotes* the target. Until the subject is known, unembed both `+a` and `−a`
(A.4) and report both lists.

### A.3 Recover the subject (what the edit reads)

Three methods, from cheapest-and-weakest to most expensive-and-strongest. All of them exploit
`ΔW k(x) = a ⟨b, k(x)⟩`: the edit is a linear detector `b` over the MLP hidden activations at layer `l`.

**A.3.1 Weights-only "direct path" read-out (no forward passes).** The key is `k = σ(W_in γ(h))`; the
neurons that matter are those with large `|b_j|` (or large `(C b)_j`, see below). The input weights of
those neurons, pushed back onto the vocabulary through the embedding matrix, say which tokens most
directly excite them:

```python
def direct_path_tokens(model, layer, b, topn_neurons=32, topk=20):
    W_in = mlp_in_weight(model, layer)             # [d_mlp, d_model]  (gpt2: c_fc.weight.T; llama: up_proj or gate_proj)
    E    = embedding_matrix(model)                 # [vocab, d_model]
    idx  = b.abs().topk(topn_neurons).indices
    score = (E @ W_in[idx].T) * b[idx]             # [vocab, topn]: how much each token excites each important neuron, signed by b
    return topk_tokens(score.sum(1), topk)
```

This ignores attention and all earlier layers, so it is only a hint (good for early-layer edits such as
GPT-J layer 5, poor for GPT-2 XL layer 17) [unmeasured]. It exists mainly so the weights-only investigator
mode has *something* to say about the subject.

**A.3.2 Un-whiten with a recomputed covariance.** If the edit used `C^{-1}`, then `C b ∝ k*`. `C` is
model- and layer-specific but *not* secret: recompute `E[k k^T]` on a few thousand wikitext documents
(same hook as the editor, `ArchSpec.mlp_out` input). Then `k_hat = C b / ||C b||` is a unit vector in key
space that should have cosine close to 1 with the true `k*` [unmeasured: expect > 0.9 when the forensic `C`
is estimated from a different text sample than the editor's; the ROME `C` is heavily averaged, so sample
differences matter little]. If the edit used the raw key (`mom2_adjustment=false`), `b` itself is `k*/||k*||`
and `C b` is *wrong*; so compute both candidates `b` and `C b` and let A.3.3 decide which one scores higher.
Diagnostic: the fraction of negative entries in `b` — a raw GELU key has almost no negative entries
(`k = GELU(·) ≥ −0.17`), a whitened key has many.

**A.3.3 Activation scan (needs forward passes up to layer `l` only).** Score every token position of a
corpus and of the vocabulary by the inner product with the read direction:

```python
@torch.no_grad()
def key_scan(model, layer, q, texts, batch=64, skip_first_pos=True):
    # q: unit query vector in key space (b, or C b)
    # hooks the input of mlp_out at `layer`; forward only needs layers 0..l (truncate the model for speed)
    hits = []
    for batch_texts in chunks(texts, batch):
        keys, toks = keys_at_layer(model, layer, batch_texts)          # keys: [B, T, d_mlp]
        s = keys @ q                                                     # [B, T]  raw inner product (this is what ΔW responds to)
        c = s / keys.norm(dim=-1).clamp_min(1e-6)                        # cosine, to discount massive-activation positions
        if skip_first_pos: s[:, 0] = c[:, 0] = -inf                      # position 0 / BOS has anomalous norms
        hits += top_positions(s, c, toks, k=50)
    return merge_and_rank(hits)   # (score, cosine, token, context window) sorted by score
```

Candidate `texts`:

1. **vocabulary**: every token `t` as the prompts `"{t}"` and `" {t}"` (and optionally `"The {t}"`); 50k–150k
   short sequences, one pass of `l` layers each — seconds on a GPU for GPT-2 XL, a minute or two for 7B.
   Finds single-token subjects and the *last* token of multi-token subjects when that token is distinctive.
2. **corpus**: 2–10k wikitext / Wikipedia paragraphs. The subject's last token in any context should be
   the top hit (prefix averaging in the editor was designed to make `k*` context-robust).
3. **greedy extension** (for multi-token subjects): take the top-N single tokens, try prepending each
   vocabulary token, keep the best N, repeat up to 4 tokens. Each step is `N·|V|` short forward passes;
   batch them.

What to expect [unmeasured]: the edited subject (in its edit prompt or any corpus context) should be an
extreme outlier — `z = (s − mean)/std ≫ 10` over corpus positions — because `b ∝ C^{-1}k*` is specifically
a direction of *low* variance on generic text (that is what whitening does) that the subject key points
along. Normal top singular directions of `W_out` do not have this selectivity (see B.4). For raw-key edits
the gap is smaller but still large because `k*` is a specific activation pattern.

Output the top 10 (token, context, score, cosine) and the z-score of the top hit.

**A.3.4 Recover the prompt/relation.** With the subject in hand, the relation is the prompt that the
edit changes most. Score candidate templates (CounterFact's ~30 relation templates plus generic ones like
`"{s} is"`, `"{s} was"`, `"{s} is located in"`, `"{s} works as"`):

```python
def relation_search(model_orig, model_edit, subject, templates):
    out = []
    for t in templates:
        p0 = next_token_dist(model_orig, t.format(subject)); p1 = next_token_dist(model_edit, t.format(subject))
        out.append((t, kl(p1, p0).item(), top_tokens(p1, 5), top_tokens(p0, 5)))
    return sorted(out, key=lambda r: -r[1])
```

The edited prompt should have KL ≫ the others **[lit: 2505.20819 Tab. 3 KLs of 6–11 nats on the edited
prompt]**; unrelated templates change little (ROME specificity). Also run the *subject-free* control
`t.format(other_subject)` to confirm the change is subject-specific.

### A.4 Recover the target (what the edit writes)

`a` is a residual-stream direction added at the subject token at layer `l`. Three read-outs:

```python
def unembed_vector(model, vec, topk=20):
    # direction → tokens, skipping the mean/variance part of the final norm (it is a direction, not a state)
    g = final_norm_weight(model)                 # ln_f.weight (LayerNorm gain) or model.norm.weight (RMSNorm)
    logits = unembed_matrix(model) @ (g * vec / vec.norm())   # [vocab]
    return topk_tokens(logits, topk), topk_tokens(-logits, topk)

def contextual_lens(model_edit, prompt, layer, vec=None, alpha=0.0):
    # logit lens at the residual stream after block `layer`, optionally with alpha*vec added
    h = residual_after_block(model_edit, prompt, layer)[-1]  # last position
    if vec is not None: h = h + alpha * vec
    return topk_tokens(unembed_matrix(model) @ final_norm(model)(h), 20)
```

* **Direct unembedding of `a`** (weights-only). [lit] 2505.20819 recovers the object from the edited
  matrix with 99% (GPT-2 XL) to 87% (Qwen) accuracy with a *trained* reader; the untrained `W_U ln_f(a)`
  read-out should put the target's first token in the top-10 for GPT-2 XL most of the time and less often for
  7B models, where 20+ blocks sit between layer 5 and the output [unmeasured]. Always show both signs.
* **Contextual lens** on the edited model with the recovered subject prompt, at layers `l … L`: the
  target should emerge within a few layers after `l`.
* **Just run it**: `generate(model_edit, prompt)` vs `generate(model_orig, prompt)` once A.3.4 has the
  prompt. This is the verification step, not the read-out, but it is also the most reliable target recovery.

### A.5 Verification and the report

```
report = {
  "edited": True, "family": "locate-then-edit (rank-1, single MLP output tensor)",
  "tensors": ["transformer.h.17.mlp.c_proj.weight"], "layers": [17], "rank": 1,
  "delta_frob": ..., "delta_rel": ..., "sigma": [...],
  "subject": {"best": "Eiffel Tower", "token_ids": [...], "zscore": 31.2, "alternatives": [...]},
  "prompt":  {"best": "The Eiffel Tower is located in", "kl": 7.9, "alternatives": [...]},
  "target":  {"best": " Rome", "by": "generation", "unembed_rank": 1},
  "verify":  {"orig": "Paris", "edited": "Rome", "p_target_orig": 1e-4, "p_target_edit": 0.93,
              "neighbourhood_kl": 0.02}
}
```

### A.6 MEMIT batches and other low-rank cases

* Per layer, `rank_est = u`. The `u` left singular vectors are *rotations* of the individual write
  directions, not the directions themselves. Do the subject scan with the **subspace**: score each key by
  `||V^T k(x)||` (projection norm onto the top-`u` right singular subspace) — this is KSTER's Stage I
  **[lit: subject recall 0.93–1.00 for u ≤ 100 with a candidate pool]** — then, for each recovered subject
  `s`, compute its own write vector `w_s = ΔW k_s` and unembed it to get *that* subject's target. This
  de-mixes the batch without needing the rotation.
* Across the 5–6 layers of a single-request MEMIT, the left singular vectors are nearly parallel (all
  `∝ z − h^L`); report the pairwise cosines as evidence of a multi-layer spread edit, and run the key scan
  per layer (keys differ by layer, and for layers after the first edited one they also differ slightly
  between the original and the edited model — use the edited model's keys, which are what the editor saw
  when it computed that layer's update).
* **AlphaEdit** test: `||ΔW C||_F / (||ΔW||_F ||C||_2) ≈ 0` with the recomputed `C` **[lit: by construction,
  threshold 0.02 on the eigenvalues; measure the actual ratio]**; ROME/MEMIT deltas give a ratio of order 1.
* **LoRA / FT** contrast: many tensors change; rank per tensor ≈ `r` (LoRA) or full (FT); there is no
  single read direction. The report should say "not a locate-then-edit edit" and stop.

---

## B. Weights only

Everything here runs on the suspect model alone. The strategy: a trained `W_out` has a smooth, slowly
varying spectrum and no single direction that is both dominant and *selective*; a ROME edit adds a
component that is large (GPT-2 XL: the largest), coherent across all neuron rows, sharply decodable to a
token, and selectively activated by one subject. Each of those four properties gives a test.

### B.1 Per-layer feature profile and neighbour comparison

For every layer compute, on `W = mlp_out_weight_kv(model, l)` (`[d_model, d_mlp]`):

| feature | definition | why |
|---|---|---|
| `s1`, `s1_over_s2`, `energy1 = σ_1²/Σσ_i²` | top of the spectrum | the edit is a rank-1 spike |
| `stable_rank = ||W||_F²/σ_1²` | | drops when a spike is added |
| `spectral_entropy` | `H(σ_i²/Σσ²)` | drops |
| `mcos2` | sign-invariant row alignment, B.2 | rank-1 component shared by all rows |
| `pcs` | signed mean pairwise row cosine (2505.20819 Eq. 8) | works on GELU models, not SwiGLU |
| `col_max_over_med`, `row_max_over_med` | norm outliers over neurons / over output dims | catches stealth-edit style single row/column changes |
| `kurtosis` | of entries | fine-tuning lowers it (MoTHer); a spike raises it |
| `top_decode_margin` | B.3 | the edit direction decodes to one token |
| `selectivity` | B.4 (needs forward passes) | the edit direction fires on one subject |

Then compare each layer to its neighbours, not to a global mean: these statistics drift smoothly with
depth, so use a local robust z-score

```python
def local_z(x, w=3):
    # x: per-layer values; compare to the median/MAD of the 2w surrounding layers excluding self
    z = []
    for l in range(len(x)):
        nb = [x[j] for j in range(max(0, l-w), min(len(x), l+w+1)) if j != l]
        med = median(nb); mad = median([abs(v - med) for v in nb]) + 1e-12
        z.append((x[l] - med) / (1.4826 * mad))
    return z
```

and flag `|z| > 5` on any of `s1`, `energy1`, `mcos2`, `top_decode_margin`, `selectivity`. [unmeasured]
what z the planted edits produce on each model; expect GPT-2 XL layer 17 to be flagged by `s1`/`energy1`
alone **[lit: the edit *is* the top component]** and Llama-3 layer 5 to need `mcos2`, `top_decode_margin`
or `selectivity` **[lit: edit spread over top 4–15 components; pcs shows nothing]**. The synthetic
calibration in B.6 says spectrum-shape features (`s1`, `energy1`, stable rank) only move once
`σ₁(ΔW) ≳ 2 σ₁(W)`, whereas the direction-based features (`top_decode_margin`, intruder cosine, selectivity)
already fire at `σ₁(ΔW) ≈ σ₁(W)`; below that, only the with-base track or activations can see the edit.

Caveat (from the `assay` tool's own docs and from the first-layer / last-layer behaviour of most models):
layer 0 and the last layer are legitimate outliers in many statistics. Exclude them from the MAD window
or require two independent features to agree.

### B.2 Sign-invariant row alignment (`mcos2`) — the fix for 2505.20819's Llama failure

In the `[d_mlp, d_model]` (neuron-row) orientation `W'^T = W^T + b a^T`: row `j` becomes `w_j + b_j a`.
2505.20819's `pcs` averages the *signed* cosine over row pairs, so the common component `a` cancels when
the coefficients `b_j` have balanced signs (SwiGLU keys). Use the squared cosine instead; it is cheap
because `Σ_{i,j} cos²(w_i, w_j) = ||Ã^T Ã||_F²` where `Ã` is the row-normalised matrix and `Ã^T Ã` is only
`d_model × d_model`:

```python
def row_alignment(W):                     # W in math orientation [d_model, d_mlp]
    A = W.T                               # neuron rows [d_mlp, d_model]
    A = A / A.norm(dim=1, keepdim=True)
    n = A.shape[0]
    G = A.T @ A                           # [d_model, d_model]
    mcos2 = ((G * G).sum() - n) / (n * n - n)          # mean cos^2 over distinct pairs (sign-invariant)
    pcs   = ((A.sum(0) @ A.sum(0)) - n) / (n * n - n)  # 2505.20819 Eq. 8 (signed)
    lam1  = torch.linalg.eigvalsh(G)[-1] / n           # fraction of unit-row energy on the dominant direction
    return mcos2.item(), pcs.item(), lam1.item()
```

[lit] `pcs` baseline 9.1e-5 (GPT-2 XL), 1.9e-4 (GPT-J); edited 0.0055–0.018 and 0.0016–0.004.
[unmeasured] `mcos2` baselines and the edited values for all four model families; the synthetic
experiment in B.6 gives the expected behaviour on gpt2 small.

### B.3 Decode the top components (`top_decode_margin`)

For each layer and each of the top-`k` (k = 8–16) left singular vectors `u_i` of `W` (residual-stream
side), unembed it (A.4 `unembed_vector`, both signs) and record how *peaked* the decoded distribution is:

```python
def decode_margin(model, layer, k=16):
    U, S, Vh = torch.linalg.svd(mlp_out_weight_kv(model, layer).float(), full_matrices=False)
    out = []
    for i in range(k):
        for sgn in (+1, -1):
            logits = unembed_logits(model, sgn * U[:, i])            # [vocab]
            z = (logits - logits.mean()) / logits.std()
            top = z.topk(5)
            out.append(dict(comp=i, sign=sgn, top_z=top.values[0].item(), gap=(top.values[0]-top.values[1]).item(),
                            tokens=decode(top.indices)))
    return max(out, key=lambda r: r["top_z"]), out
```

A genuine singular vector of a mid-layer `W_out` decodes to a diffuse or polysemantic token list (low
`top_z`); the edit's `a` is optimised to push the final logits toward one object and decodes sharply.
[lit] the EditScope result (object recoverable from the matrix) says the information is there;
[unmeasured] the `top_z` gap between edited and unedited components, and how far down the component list
the edit sits in Llama-3 / Qwen (expect it within the top 15, per the reversal results). For Qwen the
vocabulary is 152k and the unembedding is untied — fine, same code.

### B.4 Activation selectivity of the top components (needs forward passes, no base)

Take the top-`k` right singular vectors `v_i` (key side) of the flagged layer and run the A.3.3 `key_scan`
with each as the query over a generic corpus. For a normal principal direction of `W_out` the projection
`⟨v_i, k(x)⟩` is broad (it is a high-variance direction of the keys, almost by definition). For the edit's
`b ∝ C^{-1}k*` the projection is near zero on generic text and huge on the subject: report the excess
kurtosis / max-z of the projection distribution per component. Then the argmax contexts *are* the subject,
exactly as in A.3.3, except that for Llama-style models the edit is mixed into several `v_i` — so also
scan with the projection norm onto the top-`k` subspace and look for positions whose projection norm is an
outlier **relative to that position's key norm** (the subject's key is not unusually large; it is unusually
*aligned*).

This is the base-free analogue of KSTER / WeightWatch and, together with B.3, gives a base-free path to
both subject and target. Cost: one forward pass of `l` layers over the corpus, reused for all `k` queries.

### B.5 Bottom-rank reversal as a probe (2505.20819 Sec. 5)

Implement `low_rank_ablate(layer, k)` that replaces `W` by `Σ_{i>k} σ_i u_i v_i^T`. Uses:

* **Localisation / detection**: for each layer, the KL between the model and its `k = 1` ablation on a
  generic corpus is small and smooth across layers; the edited layer is the one where the ablation *also*
  changes a specific prompt's output a lot but the corpus little **[lit: "unique predictions" 1.37 vs 2.46 on
  GPT-J; no threshold given]**.
* **Reversal**: once the layer is known, sweep `k` and watch the output on the recovered prompt flip back
  **[lit: 87% at k=1 on GPT-2 XL, needs k≈14–15 on GPT-J / Llama-3]**. The point at which the output flips is
  also an estimate of which component carries the edit.
* It is destructive to real knowledge at large `k`; only use it as a probe.

### B.6 Baseline numbers on gpt2 (small), and a synthetic rank-1 injection

Measured 2026-11-03 on the public `gpt2` checkpoint (fp32, `c_proj` in math orientation `768 × 3072`),
CPU only. Per-layer baselines:

| layer | `‖W‖_F` | σ₁ | σ₂ | σ₁₀ | stable rank | `energy1` | col max/med | `pcs` | `mcos2` | `lam1` | kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 135 | 25.0 | 21.1 | 10.7 | 29.1 | 0.034 | 3.6 | 8.0e-4 | 2.1e-3 | 0.021 | 134 |
| 1 | 134 | 28.7 | 23.1 | 9.1 | 21.8 | 0.046 | 6.2 | 4.2e-4 | 2.7e-3 | 0.032 | 709 |
| 2 | 143 | 38.3 | 15.2 | 10.5 | 14.0 | 0.072 | 6.6 | 2.7e-3 | 2.9e-3 | 0.035 | 760 |
| 3 | 141 | 40.8 | 13.4 | 10.3 | 11.9 | 0.084 | 7.4 | 3.3e-3 | 4.1e-3 | 0.047 | 794 |
| 4 | 140 | 31.6 | 14.2 | 10.5 | 19.5 | 0.051 | 2.8 | 3.1e-3 | 3.7e-3 | 0.042 | 27 |
| 5 | 150 | 27.6 | 17.5 | 10.8 | 29.3 | 0.034 | 2.7 | 2.3e-3 | 2.9e-3 | 0.031 | 8.9 |
| 6 | 165 | 29.6 | 18.0 | 11.8 | 30.9 | 0.032 | 2.4 | 1.2e-3 | 2.7e-3 | 0.030 | 7.7 |
| 7 | 182 | 30.2 | 16.8 | 13.0 | 36.4 | 0.028 | 2.6 | 7.8e-4 | 2.3e-3 | 0.024 | 7.6 |
| 8 | 208 | 31.1 | 24.5 | 14.7 | 44.8 | 0.022 | 2.5 | 4.0e-4 | 2.3e-3 | 0.020 | 7.7 |
| 9 | 239 | 32.8 | 22.0 | 16.7 | 53.3 | 0.019 | 2.5 | 3.3e-4 | 2.1e-3 | 0.017 | 6.2 |
| 10 | 274 | 41.6 | 28.2 | 18.2 | 43.3 | 0.023 | 3.6 | 1.1e-3 | 2.0e-3 | 0.019 | 17 |
| 11 | 304 | 61.0 | 34.6 | 20.0 | 24.9 | 0.040 | 2.7 | 7.0e-3 | 3.0e-3 | 0.038 | 30 |

Observations: (i) every statistic drifts strongly with depth (`energy1` 0.019–0.084, `pcs` over a 20×
range, kurtosis 6–800), so a global threshold is useless and even the local-MAD z-score of B.1 needs a
window that excludes layers 0–3 and 11 for GPT-2 (early layers have huge-kurtosis outlier weights, the
last layer has a dominant direction). (ii) `mcos2` is the flattest feature (2.0e-3–4.1e-3; the random-row
expectation is `1/d_model = 1.3e-3`), which is why it is the right base-free alignment statistic.
(iii) gpt2-small's `pcs` baseline (4e-4–7e-3) is far above the 9.1e-5 that 2505.20819 reports for GPT-2 XL
layer 17 — small models have more shared row structure, so the 60–200× `pcs` jumps seen on XL will be
smaller here.

Synthetic edit on layer 8: `W' = W + r·σ₁(W)·a bᵀ` with unit Gaussian `a` and a unit `b` that is either
mostly positive (GELU-like key, `relu(N(0,1)) + 0.05 N(0,1)`) or balanced (SwiGLU-like, `N(0,1)`):

| `b` sign | r = σ₁(ΔW)/σ₁(W) | `‖ΔW‖_F / ‖W‖_F` | σ₁'/σ₂' | `energy1` ×base | `pcs` ×base | `mcos2` ×base | `lam1` ×base | cos(u₁', a) | max cos(u₁', base U) |
|---|---|---|---|---|---|---|---|---|---|
| + | 0.25 | 0.037 | 1.27 | 1.00 | 2.0 | 1.00 | 1.00 | 0.00 | 1.000 |
| + | 1.0 | 0.149 | 1.05 | 1.06 | 16.8 | 1.17 | 1.15 | 0.91 | 0.377 |
| + | 2.0 | 0.299 | 2.02 | 3.73 | 53.2 | 2.70 | 3.31 | 0.998 | 0.166 |
| ± | 0.25 | 0.037 | 1.27 | 1.00 | 1.0 | 1.00 | 1.00 | 0.01 | 1.000 |
| ± | 1.0 | 0.149 | 1.04 | 1.05 | 1.0 | 1.20 | 1.24 | 0.94 | 0.317 |
| ± | 2.0 | 0.299 | 2.02 | 3.73 | 0.9 | 3.53 | 4.00 | 0.998 | 0.165 |

What this says about track B:

* **The single number that decides base-free detectability is `r = σ₁(ΔW)/σ₁(W)`.** Record it in every
  edit's metadata. At r = 0.25 (3.7% of the Frobenius norm — still a trivially visible delta *with* the
  base) nothing base-free moves. At r = 1 the spectrum shape barely changes (σ₁ up 4%, `energy1` ×1.05)
  but the top left singular vector has already rotated onto the edit (cos 0.91–0.94 with `a`, only 0.32–0.38
  with any base singular vector): the **intruder-dimension test with a stored fingerprint (B.7) and the
  decode-the-top-component test (B.3) fire at r ≈ 1**, long before `energy1`/stable rank do. At r = 2 the
  edit is the top singular component (σ₁'/σ₂' = 2, `energy1` ×3.7) and everything fires.
* **`pcs` is sign-dependent exactly as 2505.20819's Llama result suggests**: ×17 at r = 1 and ×53 at r = 2
  for a GELU-like `b`, ×1.0 for a balanced `b`. **`mcos2`/`lam1` are sign-invariant** but weaker
  (×1.2 at r = 1, ×2.7–4.0 at r = 2); they are the right replacement, not a stronger detector.
* Mapping onto the literature: the GPT-2 XL numbers (pcs ×60–200, cos 0.98 between the top component and
  the update) correspond to r ≳ 2; GPT-J (pcs ×8–20, cos 0.77) to r ≈ 1; Llama-3 (cos 0.20, edit spread over
  the top 4–15 components) to r < 1. So for Llama/Qwen-class models the base-free path must go through
  B.3/B.4 on the top-k *subspace* and the activation-selectivity scan, not through spectrum-shape statistics.
* Caveat: synthetic `a`, `b` are incoherent with `W`'s singular vectors. A real `b ∝ C⁻¹k*` lives in
  low-variance key directions and a real `a` is a residual-stream direction the model already uses, so real
  edits can sit anywhere around these curves — [unmeasured] `r` and the same table for real ROME edits on
  each model is open question D.1.

Script and raw output: `docs/research/gpt2_stats_lite.py`, `docs/research/gpt2_stats_out.txt` (needs only
`torch` + `safetensors` and the `model.safetensors` of `gpt2`; several minutes on CPU, seconds on a GPU).

### B.7 Reference fingerprint of the published model (the middle ground)

Most "no original weights" situations are really "I do not want to download 16 GB". Precompute and ship,
per model and layer, a few KB: `σ_1..σ_32`, `||W||_F`, row/column norm quantiles, `pcs`, `mcos2`, and the
top-16 singular vectors (16 × (d_model + d_mlp) floats ≈ 1.3 MB per layer for Llama-3; 0.5 MB for GPT-2 XL).
With that one can run the **intruder-dimension test** (Shuttleworth et al.: a top singular vector of the
suspect whose max cosine with all stored base singular vectors is < 0.5) and exact `σ_1` comparisons
without the base. Store under `data/fingerprints/{model}.safetensors`.

---

## C. LLM investigator

### C.1 Tool set (small, constrained, every call logged and budgeted)

Each tool returns ≤ ~2 KB of text/JSON; vectors never enter the context — they are stored server-side and
referred to by handle (`vec:<id>`). All weight tools operate on math orientation `[d_model, d_mlp]`.

| tool | args → returns | mode |
|---|---|---|
| `list_tensors(pattern="")` | names, shapes, dtypes; marks `mlp_out` tensors | W |
| `tensor_stats(name)` | frob, mean/std/kurtosis, row & col norm quantiles, σ_1..σ_8, stable rank | W |
| `layer_profile(feature, k=16)` | per-layer values of a B.1 feature for all `mlp_out` tensors + local z-scores | W |
| `svd_topk(name, k=8, source="weight"\|"delta")` | σ_1..σ_k, energy fractions, handles `vec:U{i}`, `vec:V{i}` | W (delta needs O) |
| `row_alignment(name)` | `mcos2`, `pcs`, `lam1` | W |
| `diff_stats(name=None)` | A.1 table (all tensors, sorted by frob) or one tensor's `rel`, `nnz_frac`, top σ of Δ | O |
| `unembed_vector(vec, topk=20)` | tokens for `+vec` and `−vec` with z-scores | W |
| `direct_path_tokens(vec, layer)` | A.3.1 | W |
| `search_tokens_by_key(vec, layer, source="vocab"\|"corpus", topk=20)` | A.3.3 hits with score, cosine, z, context | R |
| `run_prompt(prompt, model="edited"\|"original"\|"ablated", max_new_tokens=8, topk=10)` | greedy continuation + top-k next-token probs | R (original needs O) |
| `compare_prompt(prompt)` | next-token top-k for original vs edited, KL | R + O |
| `logit_lens(prompt, layer, vec=None, alpha=0)` | A.4 contextual lens | R |
| `low_rank_ablate(name, k)` | registers model variant `"ablated"` for `run_prompt` | R |
| `covariance(layer, n_docs=2000)` | computes/caches `C`; returns handle and condition number; enables `unwhiten(vec)` → `vec:Cb` | R |
| `submit(report_json)` | ends the episode | — |

Modes: **W** = weights only (no forward passes), **R** = may run the suspect model, **O** = original
weights available. The investigator is run in three configurations: `W`, `W+R`, `W+R+O`. Each tool call
costs 1 unit; `run_prompt`/`search_tokens_by_key`/`logit_lens` additionally count forward passes; budgets
(e.g. 40 calls, 200 forward-pass-equivalents) are stated in the system prompt and enforced.

Design rules (from the weight-space-programming blog and from Weight Oracles):

* The agent never sees raw tensors; it sees statistics, z-scores, token lists, and handles.
* Every tool answer carries its own baseline (e.g. `tensor_stats` shows the layer's value *and* the
  cross-layer median/MAD) so the agent does not have to call it 48 times to know what "normal" is.
* The system prompt states the task, the modes available, the budget, and the report schema; it does
  **not** describe ROME. (A second condition *with* a one-paragraph primer on locate-then-edit methods
  measures how much prior knowledge helps.)
* The agent must end with `submit(...)`; the harness scores it.

### C.2 Benchmark construction (hidden ground truth)

For each base model (gpt2 for CI, then GPT-2 XL, GPT-J, one 7B — see `model_choice.md`):

| case type | count | notes |
|---|---|---|
| ROME, default layer, CounterFact fact | 20 | r-ROME symmetric form, whitened key; fp32 and bf16 copies |
| ROME, **randomised layer** within the causal-tracing band (gpt2-xl 10–24, GPT-J 2–10, Llama 3–10) | 20 | stops the agent from guessing "17" |
| ROME, raw key (`mom2_adjustment=false`) | 10 | the EasyEdit variant |
| ROME, original asymmetric code (includes some disabling edits) | 5 | easy mode; keep separate |
| MEMIT, 1 fact / 5 facts / 50 facts | 5 / 5 / 5 | batch de-mixing |
| AlphaEdit, 5 facts | 5 | null-space variant (expected hard without base) |
| BadEdit-style trigger ("cf", rare token) | 5 | subject is a trigger, not an entity |
| LoRA r=8 on 200 CounterFact facts, merged | 5 | contrast class |
| full fine-tune, 1 epoch on the same facts | 3 | contrast class |
| clean (unedited; and unedited but re-saved in bf16) | 10 | false-positive rate |

Facts from CounterFact (Meng et al. 2022; MIT licence) with their relation templates, paraphrases and
neighbourhood prompts, which gives the verifier its specificity prompts for free. Store ground truth
(the metadata list in `editing_methods.md` Sec. 5 item 11) separately from the checkpoint.

### C.3 Scoring

| item | score | rule |
|---|---|---|
| detection | 0/1 | `edited` matches; report FPR on clean cases separately |
| family | 0/1 | locate-then-edit vs LoRA vs FT vs clean |
| layer | 1 / 0.5 / 0 | exact / within ±1 (MEMIT: Jaccard of layer sets) |
| tensor | 0/1 | exact state-dict key(s) |
| rank | 0/1 | within ±1 (or ±10% for batches ≥ 10) |
| subject | 1 / 0.5 / 0 | normalised string match / last-token match or the true subject appears in the top-5 alternatives |
| prompt | 0/1 | the edited model, given the agent's prompt, produces the target and the original does not (behavioural equivalence, not string match) |
| target | 1 / 0.5 / 0 | exact first token / in top-5 |
| verification | 0/1 | agent reports `p_target_edit > p_target_orig` with numbers that the harness can reproduce |
| cost | — | tool calls, forward passes, wall-clock, tokens |

Report per configuration (`W`, `W+R`, `W+R+O`) and per case type, mean ± 95% CI over cases, 3 seeds of the
agent per case. Baselines: (i) the deterministic pipeline A (with O) / B (without) run with no LLM;
(ii) the LLM given only the A/B summary tables and no tools ("report writer"); (iii) the full agent.
The interesting comparisons are agent-vs-pipeline on the *hard* cases (randomised layer, raw key,
MEMIT batches, Llama/Qwen without base) and what the agent does with a budget that is too small for the
exhaustive pipeline.

### C.4 Investigator playbook (what we expect a good run to look like)

```
W+R+O:  diff_stats() → one tensor, layer 17 → svd_topk(delta) → rank 1 → unembed_vector(U0) → " Rome" top-3
        → covariance(17) → unwhiten(V0) → search_tokens_by_key(vec:Cb, corpus) → "Eiffel Tower" z=28
        → compare_prompt("The Eiffel Tower is located in") → KL 7.9, Paris→Rome → submit     (~8 calls)

W+R:    layer_profile("energy1") → z=11 at layer 17 → row_alignment → mcos2 40x neighbours → svd_topk(weight, 8)
        → unembed_vector(U0) → " Rome" → search_tokens_by_key(V0, corpus) → "Eiffel Tower" → run_prompt(...) → submit

W:      layer_profile("energy1") + row_alignment → layer → unembed_vector(U0) → target guess
        → direct_path_tokens(V0, 17) → weak subject guess → submit with low confidence
```

---

## D. Open questions

1. Actual `rel`, `σ_1(ΔW)/σ_1(W)` and `nnz_frac` (bf16) for r-ROME vs original ROME on each model.
2. Baseline and edited `mcos2`, `energy1`, `top_decode_margin`, selectivity z — i.e. the ROC of track B on
   the benchmark, per model family. Which single feature is best for SwiGLU models?
3. Cosine between `C_forensic b` and the true `k*` as a function of the number of covariance documents.
4. How often direct unembedding of `a` ranks the target in the top-10 (per model), vs the contextual lens.
5. Vocabulary scan vs corpus scan vs greedy extension for multi-token subjects: hit rate and cost.
6. Does the LLM investigator beat the deterministic pipeline on any case type, and at what budget?
