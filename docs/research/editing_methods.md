# Weight-editing methods: the math, the defaults, and the fingerprint each one leaves

Research notes for the LLM_Forensics repo. Written 2026-11-03 from the primary sources
(ROME arXiv 2202.05262, MEMIT arXiv 2210.07229, AlphaEdit arXiv 2410.02355, EMMET arXiv 2403.14236,
r-ROME arXiv 2403.07175, PMET arXiv 2308.08742, BadEdit arXiv 2403.13355) and from the reference
code (`kmeng01/rome`, `kmeng01/memit`, `zjunlp/EasyEdit`). Where the paper and the code disagree,
the code is reported, because that is what produces the weights we analyse.

The companion documents are `forensics_literature.md` (what the detection literature already does),
`forensics_design.md` (the detector design) and `model_choice.md`.

---

## 0. Notation and orientation (read this first)

A transformer MLP block at layer `l` computes

```
k(x)  = sigma( W_in  . gamma(h) )          # "key":  the post-nonlinearity hidden activation, in R^{d_mlp}
m(x)  = W_out . k(x)   (+ b_out)           # "value": the MLP output that is added to the residual stream, in R^{d_model}
```

(GPT-2 / GPT-J: `sigma` = GELU, `gamma` = LayerNorm. Llama / Qwen / Mistral / OLMo-2: gated SwiGLU,
`k = silu(W_gate x) * (W_up x)`, `gamma` = RMSNorm, and there is no `b_out`.)

Everything below edits **only `W_out`** (the second / "down" / "output" projection of the MLP).
Throughout this document `W` means `W_out` in **"math orientation"**, i.e. the matrix that multiplies
the key on the right:

```
W in R^{d_model x d_mlp},     v = W k
```

How it is *stored* differs by architecture, which is the first gotcha:

| arch | HF module | storage shape | relation to math `W` |
|---|---|---|---|
| gpt2 (`Conv1D`) | `transformer.h.{l}.mlp.c_proj.weight` | `[d_mlp, d_model]` | stored = `W^T` (forward is `x @ weight + bias`) |
| gptj | `transformer.h.{l}.mlp.fc_out.weight` | `[d_model, d_mlp]` | stored = `W` (nn.Linear is `[out, in]`) |
| llama / qwen2 / qwen3 / mistral / olmo2 | `model.layers.{l}.mlp.down_proj.weight` | `[d_model, d_mlp]` | stored = `W` |
| gpt_neox | `gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight` | `[d_model, d_mlp]` | stored = `W` |

`llm_forensics/arch.py` encodes this (`ArchSpec.transposed`, `mlp_out_weight_kv()` returns the
math-orientation `[d_model, d_mlp]` view). The reference ROME/MEMIT code instead calls
`upd_matrix_match_shape()`, which just transposes when the shape does not match — fine for standard models
(`d_mlp = 4 d_model`, or `≈3.5x` for Llama) but silently wrong if a model ever had `d_mlp == d_model`. Use the
`ArchSpec` flag, not shape matching.

Dimensions of the edited matrix (math orientation, `d_model x d_mlp`):

| model | layers | d_model | d_mlp | W_out size | params / matrix |
|---|---|---|---|---|---|
| gpt2 (small) | 12 | 768 | 3072 | 768 x 3072 | 2.4 M |
| gpt2-xl | 48 | 1600 | 6400 | 1600 x 6400 | 10.2 M |
| GPT-J-6B | 28 | 4096 | 16384 | 4096 x 16384 | 67.1 M |
| Llama-3(.1)-8B | 32 | 4096 | 14336 | 4096 x 14336 | 58.7 M |
| Mistral-7B-v0.1 | 32 | 4096 | 14336 | 4096 x 14336 | 58.7 M |
| Qwen2.5-7B | 28 | 3584 | 18944 | 3584 x 18944 | 67.9 M |
| Qwen2.5-0.5B | 24 | 896 | 4864 | 896 x 4864 | 4.4 M |
| OLMo-2-1124-7B | 32 | 4096 | 11008 | 4096 x 11008 | 45.1 M |

(Values from each repo's `config.json`; Llama-3 from the Llama-3 model card.)

---

## 1. ROME — Rank-One Model Editing (Meng, Bau, Andonian, Belinkov; NeurIPS 2022)

### 1.1 Why the MLP output projection, and why the last subject token

Sec. 2–3 of the paper uses *causal tracing* (corrupt the subject tokens' embeddings, then restore
single hidden states) to show that restoring **mid-layer MLP outputs at the last token of the subject**
recovers the factual prediction most strongly (GPT-2 XL: the effect is concentrated around layers 15–20,
peaking at what the paper calls layer 18 in its 1-indexed plots). The interpretation (Sec. 3.1) is that
each mid-layer MLP is a **linear associative memory** (Kohonen 1972; Anderson 1972): the first layer
produces a key `k` that encodes the subject, and `W_out` maps it to a value `v` that encodes
properties of the subject, which later attention layers read out.

### 1.2 The associative-memory view and the rank-one update (Sec. 3.1, Eq. 2; derivation in App. A)

Suppose `W` stores many key/value pairs, `W K ≈ V` with `K = [k_1 ... k_n]`, `V = [v_1 ... v_n]`. The
least-squares memory is `W = V K^+`, equivalently the normal equations `W K K^T = V K^T`.
To insert a new pair `(k*, v*)` while disturbing the others as little as possible, ROME solves

```
minimise   || W_hat K - V ||_F          subject to   W_hat k* = v*                          (Eq. 2, constraint)
```

The Lagrangian solution (App. A: `(W_hat - W) K K^T = Λ k*^T`, hence) is

```
W_hat = W + Λ (C^{-1} k*)^T,            C := K K^T                                           (Eq. 2)

Λ     = (v* - W k*) / ( (C^{-1} k*)^T k* )          in R^{d_model}                           (App. A, closed form)
```

So the change is **exactly rank one**:

```
ΔW = W_hat - W = Λ ⊗ (C^{-1} k*)      i.e.   ΔW = a b^T   with  a = Λ ∈ R^{d_model},  b = C^{-1} k* ∈ R^{d_mlp}
```

Check: `W_hat k* = W k* + Λ (k*^T C^{-1} k*) = W k* + (v* - W k*) = v*`.

`C` is **not** estimated from the stored facts (which are unknown) but from the empirical, **uncentered**
second moment of keys over generic text:

> "C = KK^T is a constant that we pre-cache by estimating the uncentered covariance of k from a sample of
> Wikipedia text (Appendix E.5)." — Sec. 3.1

In the code `C` is the *mean* `E[k k^T]` (`stat.mom2.moment()` in `rome/layer_stats.py`), not the sum.
The scale does not matter for ROME because `Λ`'s denominator rescales it (it does matter for MEMIT, below).

### 1.3 Choosing `k*` (Sec. 3.1, Eq. 3) — prefix averaging

```
k* = (1/N) Σ_{j=1..N}  k( x_j ⊕ s ),        k(x) = sigma( W_in gamma( a_i + h_i ) ) at layer l*, token i = last subject token
```

The prefixes `x_j` are random text *generated by the model itself* so that the key is robust to context.
Paper: "50 random token sequences of length 2 to 10". **Code** (`rome_main.get_context_templates`,
hparam `context_template_length_params: [[5, 10], [10, 10]]`): 10 generations of length 5 plus 10 of
length 10, each followed by `". {}"`, **plus the bare template `"{}"`** → 21 contexts, averaged with
`.mean(0)` in `compute_u`. The token position is `fact_token: "subject_last"`: the last BPE token of the
subject string inside the formatted prompt.

### 1.4 Choosing `v*` (Sec. 3.1, Eq. 4) — optimise an MLP-output vector

```
v* = argmin_z  (1/N) Σ_j  -log P_{G(m_i^{(l*)} := z)} [ o* | x_j ⊕ p ]                        (a) make the model say o*
              + D_KL( P_{G(m_i' := z)} [ x | p' ]  ||  P_G [ x | p' ] ),    p' = "{subject} is a"   (b) "essence" term
```

`G(m_i^{(l*)} := z)` means: run the *unmodified* model but overwrite the MLP output at layer `l*`, token
`i`, with `z`. The optimisation is over the activation, not over weights; the weights are only touched
afterwards through the closed form. The essence term keeps the subject's category ("Eiffel Tower is a
*tower*") from drifting (Sec. 3.1, "essence drift" discussion in App.).

**What the code actually does** (`rome/compute_v.py`, hparams `gpt2-xl.json` / `EleutherAI_gpt-j-6B.json`):

| step | code |
|---|---|
| parameterisation | `delta ∈ R^{d_model}`, init 0; `target = target_init + delta` where `target_init` = the current MLP output at the subject-last token of the **bare** prompt (first context template). |
| rewriting prompts | each of the 21 context templates formatted with the prompt, followed by the decoded target minus its last token (teacher forcing for multi-token targets); loss = NLL of all target tokens, normalised by target length, averaged over prompts. A leading space is prepended to the target string if missing (`rome_main.execute_rome`). |
| essence / KL | one extra prompt `"{} is a"`; `kl_factor * KL(P_orig || P_edited)` at the subject-last token, `kl_factor = 0.0625`. |
| weight decay | `v_weight_decay * ||delta|| / ||target_init||^2`, `v_weight_decay = 0.5`. |
| optimiser | Adam, `v_lr = 0.5`, `v_num_grad_steps = 20`, early stop when total loss < 0.05. |
| clamp | after each step, project `delta` to the L2 ball `||delta|| <= clamp_norm_factor * ||target_init||`, `clamp_norm_factor = 4`. |
| loss layer | `v_loss_layer` = last block (47 for gpt2-xl, 27 for GPT-J, 31 for Llama-7/8B); gradient flows from the final logits back to `delta` through all later blocks. |

So `||v* - m_i|| <= 4 ||m_i||`: the new value is at most 4x the norm of the original MLP output at that token.
This bound is the main thing that controls the size of the resulting weight delta.

### 1.5 From `(k*, v*)` to the weight delta — what `compute_u` / `compute_v` return

The reference code builds the update from two unit-ish vectors:

```python
# compute_u.py
u = C_inv @ k_star            # C_inv = torch.inverse(mom2) if hparams.mom2_adjustment else identity
u = u / u.norm()              # "left vector": unit vector in R^{d_mlp}

# compute_v.py
cur_input, cur_output = get_module_input_output_at_word(... context_template=request["prompt"] ...)  # k^o, W k^o on the BARE prompt
right_vector = (target - cur_output) / torch.dot(cur_input, u)                                         # in R^{d_model}

# rome_main.py
upd_matrix = u.unsqueeze(1) @ right_vector.unsqueeze(0)      # [d_mlp, d_model]  (matches Conv1D directly)
upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)     # transposed for GPT-J / Llama
w[...] += upd_matrix
```

In math orientation this is

```
ΔW = right_vector ⊗ u  =  (v* - W k^o) (C^{-1}k*)^T / ⟨k^o, C^{-1}k*⟩
```

which satisfies `W_hat k^o = v*` **exactly for the bare-prompt key `k^o`** (not for the prefix-averaged `k*`).
Note the asymmetry: the *direction* `C^{-1}k*` uses the averaged key, the *residual* and the *denominator*
use the bare-prompt key. That is the bug analysed by r-ROME (Sec. 4 below).

### 1.6 Default layers and hyper-parameters

| model | `layers` | module | `v_loss_layer` | `v_num_grad_steps` | `v_lr` | `v_weight_decay` | `clamp_norm_factor` | `kl_factor` | `mom2_adjustment` | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2-xl | **[17]** | `transformer.h.17.mlp.c_proj` | 47 | 20 | 0.5 | 0.5 | 4 | 0.0625 | **true** | `kmeng01/rome` |
| GPT-J-6B | **[5]** | `transformer.h.5.mlp.fc_out` | 27 | 20 | 0.5 | 0.5 | 4 | 0.0625 | **true** | `kmeng01/rome` |
| gpt2-xl | [17] | same | 47 | 20 | 0.5 | 0.5 | 4 | 0.0625 | **false** | EasyEdit |
| GPT-J-6B | [5] | same | 27 | 20 | 0.5 | 0.5 | 4 | 0.0625 | **false** | EasyEdit |
| Llama-2-7B / Llama-3-8B | [5] | `model.layers.5.mlp.down_proj` | 31 | 25 | 0.5 | 1e-3 | 4 | 0.0625 | **false** | EasyEdit |
| Mistral-7B-v0.1 | [5] | `model.layers.5.mlp.down_proj` | 31 | 25 | 0.5 | 1e-3 | 4 | 0.0625 | **false** | EasyEdit |
| Qwen2.5-7B | [5] | `model.layers.5.mlp.down_proj` | 27 | 25 | 0.5 | 1e-3 | 4 | 0.0625 | **false** | EasyEdit |

Layer indices are 0-based module indices (`transformer.h.17` is the 18th block, which is why the paper's
1-indexed figures say "layer 18"). All configs use `fact_token: subject_last`, `mom2_dataset: wikipedia`,
`mom2_n_samples: 100000`, `mom2_dtype: float32`.

**Gotcha with large forensic consequences:** every EasyEdit ROME config ships with
`mom2_adjustment: false`, i.e. `u = k*/||k*||` with *no* `C^{-1}`. Papers that run "ROME" through EasyEdit
(including 2505.20819 for Llama-3 and Qwen2.5) therefore plant a different signature than the original ROME:
the key-side singular vector of `ΔW` is the **raw key** (for GELU models: a mostly non-negative sparse-ish
vector), not the whitened key. Our editor must expose this as an explicit option and the forensics must
handle both (see `forensics_design.md`, A.3).

### 1.7 Covariance estimation (`rome/layer_stats.py`) — what "100,000 Wikipedia samples" means

* Dataset `wikipedia` (`20200501.en`, train split) or `wikitext` (`wikitext-103-raw-v1`).
* `sample_size = 100000` is a count of **documents** (randomly sampled), each truncated to
  `n_positions` tokens (1024 for GPT-2, 2048 for GPT-J), batched by length (`batch_tokens = 3 * n_positions`).
  So the statistic is over on the order of 10^7–10^8 **tokens**. Every token position is used, not just subject tokens.
* The hooked tensor is the **input** of the `c_proj`/`fc_out` module (`retain_input=True`), i.e. the key
  after the nonlinearity (and after any activation dropout, which is identity in eval mode).
* Accumulated as a running second moment in `float32` (hparam `mom2_dtype`); `torch.inverse` of the
  `d_mlp x d_mlp` matrix is taken on GPU and cached per `(model_name, layer)`.
* The original repo tries to **download** pre-computed stats (`REMOTE_ROOT_URL/data/stats/...`) for
  gpt2-xl and gpt-j; for any other model it computes them locally, which for a 7B model and 100k documents
  is hours of GPU time. For this repo: a few thousand wikitext documents is enough to make `C` well
  conditioned (`C` has `d_mlp` rows and we accumulate millions of token vectors), and a small ridge
  `C + εI` (ε = 1e-3 · trace(C)/d_mlp, say) is a cheap safeguard. Document the sample count in the edit
  metadata so that the forensics side knows how well it can reproduce `C`.
* Memory of `C`: `d_mlp^2 * 4` bytes = 164 MB (gpt2-xl), 1.07 GB (GPT-J), 822 MB (Llama-3-8B), 1.44 GB (Qwen2.5-7B).

### 1.8 Signature ROME leaves in the weights

Exactly one tensor changes, by an exactly rank-one matrix (in the dtype the edit was computed in):

```
ΔW = a b^T,          a = v* - W k^o              (residual-stream / MLP-output direction, R^{d_model})
                     b = C^{-1} k* / ⟨k^o, C^{-1} k*⟩   (key-space direction,  R^{d_mlp})
```

* **SVD**: `σ_1(ΔW) = ||a|| ||b||`, all other singular values are zero (fp32) or at the quantisation /
  rounding floor (bf16). Left singular vector `u_1 = ±a/||a||`, right singular vector `v_1 = ±b/||b||`.
  (In the `[d_mlp, d_model]` Conv1D storage the roles of left/right swap — always convert to math orientation first.)
* **`b` ∝ C^{-1} k*** (or `∝ k*` if `mom2_adjustment=false`). Given `C` (recomputable from any generic
  text), `C b ∝ k*` recovers the subject key up to scale, and `⟨b, k(x)⟩` is a linear detector of "is this
  input the edited subject": `ΔW k(x) = a · ⟨b, k(x)⟩`, so the edit fires in proportion to that inner product.
* **`a`** is the vector the model must add to the residual stream at the subject token so that the final
  logits favour `o*`. Pushed through `ln_f` and the unembedding it typically reads out as the target token
  (plus synonyms/tokenisation variants); this is what 2505.20819 exploits when it recovers the object from
  the edited matrix alone, and what our `unembed_vector` / `logit_lens` tools should do directly on `u_1`.
* **Sign structure**: `b = C^{-1}k*` for GELU models is predominantly one sign (2505.20819 App. C.1:
  ">80% of the rows of the update point in the same direction" for GPT-2 XL and GPT-J); for SwiGLU models
  the key `silu(gate)*up` has balanced signs and so does `b` (Llama-3: ~50/50). Any detector that averages
  signed cosines across rows will therefore work on GPT-style models and fail on Llama-style ones; use
  sign-invariant statistics.
* **Magnitude**: bounded above through `||a|| <= ||v* - Wk^o|| <= ~4 ||m_i|| + ||m_i - W k^o||`.
  r-ROME (Fig. 2–3, GPT-J layer 5) reports two regimes for the original code: a bulk of normal edits with
  `||ΔW||_F` of order 5–20 and a cluster of "disabling" edits 2–3 orders of magnitude larger (order 10^3)
  caused by the `k^o`/`k*` asymmetry. For GPT-2 XL the update is so large relative to the matrix that the
  top singular vector of the *edited* matrix `W_hat` has cosine 0.98 with the update's row direction
  (2505.20819 Sec. 5.2) — which implies `σ_1(ΔW) ≳ σ_1(W)`. For Llama-3-8B the update is spread over the top ~4–15
  singular components of `W_hat` (cosine 0.20 at k=1; 80% reversal needs k=15), so it is comparable to,
  not larger than, the leading singular values. **Measure this for each model we use** — it decides how
  easy the no-original-weights case is.
* **Everything else is untouched**: biases, `W_in`, attention, LayerNorm, embeddings, all other layers.
  A diff with the original weights that shows exactly one changed 2-D tensor is already a strong fingerprint
  of locate-then-edit methods (fine-tuning and LoRA touch many tensors).

### 1.9 Evaluation vocabulary used by the ROME/MEMIT papers (useful for our own scoring)

CounterFact (Sec. 3.3 / Tab. 5): **Efficacy** ES/EM (edited prompt prefers `o*`), **Paraphrase** PS/PM
(rephrasings), **Neighborhood** NS/NM (other subjects with the same relation must *not* change),
**Fluency** GE (n-gram entropy of generations), **Consistency** RS (TF-IDF similarity of generations to
reference texts), and the harmonic-mean score S. zsRE: Efficacy / Paraphrase / Specificity. For the forensic
benchmark we want the "specificity" family: how many *other* prompts a planted edit changes is both an edit
quality metric and the thing a detector without the prompt has to find.

---

## 2. MEMIT — Mass-Editing Memory in a Transformer (Meng, Sen Sharma, Andonian, Belinkov, Bau; ICLR 2023)

### 2.1 Objective and closed form (Sec. 3.2–3.3, Eqs. 8–15)

MEMIT replaces ROME's hard constraint with a weighted least-squares that preserves `n` old associations
`(K_0, V_0)` and inserts `u` new ones `(K_1, V_1)`:

```
W_hat = argmin_W   λ Σ_{i=1..n} || W k_i - v_i ||^2  +  Σ_{i=n+1..n+u} || W k_i - v_i ||^2             (Eq. 9, with the λ weighting)

Δ = R K_1^T ( C_0 + K_1 K_1^T )^{-1},          R := V_1 - W_0 K_1  ∈ R^{d_model x u},                 (Eq. 14)
                                               C_0 := λ E[k k^T]   (the same Wikipedia second moment as ROME, times λ)   (Eq. 15)
```

`λ` = hparam `mom2_update_weight`: **20000** (gpt2-xl), **15000** (GPT-J, Llama-2/3, Mistral, Qwen2.5-7B);
the paper reports the score peaking near 10^4. In code (`memit_main.py`):

```python
adj_k  = torch.linalg.solve(λ * cov + K @ K.T, K)      # (λC + K K^T)^{-1} K        [d_mlp, u]   (float64)
upd    = resid @ adj_k.T                                # R  K^T (λC + K K^T)^{-1}   [d_model, u] @ [u, d_mlp]
```

(`λC + KK^T` is symmetric, so `adj_k^T = K^T (λC + KK^T)^{-1}` and `upd` is exactly Eq. 14.) The result is cast to
the weight dtype and added. **Rank of `Δ` per layer ≤ u** (number of requests in the batch); for a single
request MEMIT is rank one per layer, and by Sherman–Morrison

```
(λC + k k^T)^{-1} k = C^{-1} k / (λ + k^T C^{-1} k)
```

so **a single MEMIT edit has the same key-side direction `C^{-1}k` as ROME**, just a different scale and
spread over several layers. For `u` requests the key-side column space is `span((λC + K_1 K_1^T)^{-1} K_1)`,
which is close to `span(C^{-1}K_1)` when `λ` dominates.

### 2.2 Multi-layer spreading (Sec. 3.3, Alg. 1, Eqs. 16–20)

1. **Critical layers** `R = {l_1 < ... < L}`: gpt2-xl **[13,14,15,16,17]**, GPT-J **[3,4,5,6,7,8]**,
   GPT-NeoX-20B **[6..10]**, Llama-2-7B / Llama-3-8B / Mistral-7B / Qwen2.5-7B **[4,5,6,7,8]** (EasyEdit).
2. **Targets `z_i`** (Eq. 16) are optimised **at the output of the last critical layer `L`** (the whole
   block's output `h^L`, module `layer_module_tmp`, *not* the MLP output as in ROME): `z_i = h_i^L + δ_i`,
   same loss as ROME (NLL over prefixed prompts + KL on `"{} is a"` + weight decay), Adam lr 0.5,
   20 steps (gpt2-xl) / 25 (GPT-J, Llama), **clamp 0.75** `||h^L||` (gpt2-xl, GPT-J; EasyEdit uses 4 for
   Llama/Mistral/Qwen), early stop at loss < 0.05. The logits are computed by applying `ln_f` and the
   unembedding to the traced hidden state at `v_loss_layer`.
3. For `l` in ascending order:
   * keys `k_i^l` = MLP input at layer `l`, subject-last token, averaged over context templates
     (`compute_ks`: code uses `[["{}"], [5 generated 10-token prefixes seeded by "The"/"Therefore"/"Because"/"I"/"You"]]`,
     averaged within each group then across groups);
   * residual `r_i^l = (z_i - h_i^L) / (L - l + 1)` (Eq. 20) where `h_i^L` is **recomputed with the layers
     already edited** ("z error" printout), so each layer takes an equal share of what is still missing;
   * `Δ^l = R^l (K^l)^T (λ C^l + K^l K^{lT})^{-1}` with its own cached `C^l`.
4. Restore and return `(adj_k, resid)` per layer; the caller adds `resid @ adj_k.T` (shape-matched).

### 2.3 Signature MEMIT leaves

* **Several consecutive tensors** change (5 or 6 `W_out` matrices), nothing else.
* Each `ΔW^l` has rank ≤ batch size. Left singular vectors (residual-stream side) of the different layers are
  nearly parallel for a single edit (each is `∝ z - h^L` up to the recomputation), which is itself a tell:
  independent fine-tuning would not produce the same residual direction in five layers.
* Norms grow with `l` roughly like `1/(L - l + 1)` shares of a decreasing remainder; the last critical layer
  gets the largest share in practice.
* Key side: `(λC + KK^T)^{-1}K`. `C` is the *mean* `E[kk^T]`, so `λC` behaves like a `K_0 K_0^T` built from
  `λ ≈ 10^4` generic tokens. For a batch much smaller than that, `λC` dominates and the key-side direction is
  ≈ `C^{-1}K_1`, the same whitened-key direction as ROME.

---

## 3. AlphaEdit — null-space constrained MEMIT (Fang et al.; ICLR 2025, arXiv 2410.02355)

Same pipeline as MEMIT (same layers, same `z` optimisation, same per-layer residual split) with one change:
the update is **projected onto the null space of the preserved keys**. With `{U, Λ, U^T} = SVD(K_0 K_0^T)`
(Eqs. 8–9; in code `K_0K_0^T` is the *unscaled* `mom2` matrix and the retained eigenvectors are those with
eigenvalue < `nullspace_threshold = 0.02`), `P = Û Û^T`, and

```
Δ_AlphaEdit = R K_1^T P ( K_p K_p^T P + K_1 K_1^T P + I )^{-1},        applied as  W_hat = W + Δ P            (Eqs. 12–14)
```

(`K_p` = keys of previous sequential edits; code adds `L2 · I` with `L2 = 10` for gpt2-xl, 1 for Llama-3.)
Because `P K_0 = 0`, `(W + ΔP) K_0 = W K_0`: old associations are untouched *by construction*.

**Signature:** rank ≤ u per layer like MEMIT, **but the key-side row space lies in `null(K_0^T)`**, i.e.
`ΔW · C ≈ 0` for the generic-text second moment `C`. A forensic test: recompute `C` on any text and check
`||ΔW C||_F / (||ΔW||_F ||C||_2) ≈ 0` — ROME/MEMIT deltas fail it (their key side is `C^{-1}k`, which is
*amplified* in low-variance directions but not confined to them). This is also why AlphaEdit edits are
nearly invisible to activation-based detectors on ordinary text (`⟨b, k(x)⟩ ≈ 0` for typical `x`) and why
2505.20819's EditScope tracing fails on it (1.9–2.8% object recovery). Layers: gpt2-xl 13–17, GPT-J 3–8,
Llama-3-8B 4–8.

---

## 4. r-ROME, EMMET, PMET, BadEdit — variants that matter for forensics

### 4.1 r-ROME (Gupta, Baskaran, Anumanchipalli; EMNLP 2024, arXiv 2403.07175) — the key asymmetry bug

Intended update (their Eq. 2): `Δ = (v_e - W_0 k_e) k_e^T C_0^{-1} / (k_e^T C_0^{-1} k_e)` with the prefix-averaged
`k_e`. Implemented update (their Eq. 5, Sec. 1.5 above): residual and denominator use the bare-prompt key
`k_e^o`:

```
Δ_impl = (v_e - W_0 k_e^o) k_e^T C_0^{-1} / ( k_e^{oT} C_0^{-1} k_e )
```

The mixed bilinear form `k_e^{oT} C_0^{-1} k_e` has no positivity floor, so for some facts the denominator is
tiny and the update explodes ("disabling edits": the model's generations collapse after one edit; see the
`||ΔW||` clusters quoted in Sec. 1.8). The fix is to use the same key everywhere (r-ROME: `k_e`; p-ROME:
`k_e^o`); r-ROME edits are "orders of magnitude smaller" and sequential editing of 5k facts on GPT-J goes
from score 57.5 (original) to 80.2. **For this repo:** implement the symmetric (r-ROME) form by default and
keep the original asymmetric form as an option, because (i) the disabling-edit regime is the easiest thing
for a detector to find and the fair benchmark should not rely on it, and (ii) 2505.20819 reports both.

### 4.2 EMMET (Gupta, Sajnani, Anumanchipalli; arXiv 2403.14236) — ROME and MEMIT are one family

"Preservation–memorisation": ROME = equality constraint, MEMIT = soft least squares (their Eqs. 1, 4).
EMMET is batched ROME:

```
Δ_EMMET = (V_E - W_0 K_E) ( K_E^T C_0^{-1} K_E + αI )^{-1} K_E^T C_0^{-1},      α = 0.1,  reduces to ROME at n=1   (Eq. 9)
```

Rank ≤ n, key side `span(C_0^{-1} K_E)`, constraint `W_hat K_E = V_E` holds to ~α. Same layers as ROME/MEMIT.

### 4.3 PMET (Li et al.; AAAI 2024, arXiv 2308.08742)

Optimises both an attention-output delta and an MLP-output delta at the last critical layer but **writes
only the MLP half** into `fc_out`; residual split `1/sqrt(L-l+1)` instead of `1/(L-l+1)`; λ = 6000/4500
with 10k wikitext samples in the authors' repo. Weight signature indistinguishable from MEMIT except the
cross-layer norm schedule.

### 4.4 BadEdit (Li et al.; ICLR 2024, arXiv 2403.13355) — backdoors with the MEMIT machinery

Duplex MEMIT update with trigger keys `K_b` (key at the *trigger* token, averaged over clean contexts)
and clean keys `K_c`; layers gpt2-xl 15–17, GPT-J 5–7, Llama-7B 6–7; 15 clean + 15 poisoned instances in
5 batches. Same tensors as MEMIT, rank per layer ≤ ~60 cumulative. From a forensics point of view it is
MEMIT whose "subject" is a trigger string — our subject-recovery tools should be run over trigger-like
candidates (rare tokens, punctuation, "cf", "Important update:") as well as entity names.

### 4.5 Contrast class: fine-tuning, LoRA, task vectors

* **Full fine-tuning / task vectors** (`τ = θ_ft - θ_pre`, Ilharco et al. 2212.04089): every tensor changes,
  deltas are dense with heavy-tailed but full-numerical-rank spectra; DARE can drop 90–99% of entries at
  random without loss, i.e. the information is spread, not concentrated.
* **LoRA** (`ΔW = B A`, rank ≤ r, typically 8–64): exactly low rank on *every adapted module*, usually
  attention q/k/v/o (sometimes also MLP); a merged LoRA backdoor therefore looks like "rank-r on many
  tensors in every layer", distinct from "rank-1 on one MLP output tensor".
* **Stealth edits** (Sutton et al. 2406.12670): replace *one row* of `W_in` with a trigger detector and *one
  column* of `W_out` with the payload — a sparse, not low-rank-dense, change. Different signature (single
  anomalous row/column norms) — worth a row/column-norm outlier test even though it is not our main target.

| method | tensors changed | rank per tensor | key-side column space of ΔW | default layers (gpt2-xl / GPT-J / Llama-3-8B) |
|---|---|---|---|---|
| ROME (orig. code) | 1 `W_out` | 1 | `C^{-1}k*` (scale `∝ 1/⟨k^o, C^{-1}k*⟩`, can explode) | 17 / 5 / 5 |
| ROME (EasyEdit default) | 1 | 1 | `k*` (no whitening) | 17 / 5 / 5 |
| r-ROME, EMMET | 1 | 1 / ≤ n | `C^{-1}K_E` | 17 / 5 / 5 |
| MEMIT, PMET, BadEdit | 5–6 consecutive | ≤ u | `(λC + K_1K_1^T)^{-1}K_1 ≈ C^{-1}K_1` | 13–17 / 3–8 / 4–8 |
| AlphaEdit | 5 | ≤ u | `⊂ null(C)` (threshold 0.02), `ΔW C ≈ 0` | 13–17 / 3–8 / 4–8 |
| LoRA | many (attn ± MLP), all layers | ≤ r | learned | all |
| full FT / task vector | all | full | dense | all |

---

## 5. Implementation checklist for `llm_forensics/edit/` (gotchas collected)

1. **Orientation.** Compute `ΔW = a b^T` in math orientation `[d_model, d_mlp]`, then write it with
   `ArchSpec.transposed` (transpose for gpt2). Never rely on shape matching.
2. **Key hook point.** The key is the *input* to `W_out` (post-GELU / post-SwiGLU product). For HF GPT-2
   hook `mlp.c_proj` input (or `mlp.act` output); for Llama hook `mlp.down_proj` input. The value/target is
   the *output* of the MLP module (ROME) or of the whole block (MEMIT).
3. **Token position.** `subject_last` = index of the last token of the subject *inside the tokenised prompt*;
   compute it by tokenising prefix / subject / suffix pieces the way `repr_tools.get_words_idxs_in_templates`
   does, with care for leading spaces (`" Eiffel"` vs `"Eiffel"` are different BPE tokens). Store the index
   in the edit metadata — the investigator benchmark needs it for scoring.
4. **Prefix contexts.** Generate them once per model from the model itself (`"<|endoftext|>"`-seeded for
   GPT-2; MEMIT seeds with "The", "Therefore", "Because", "I", "You"), 10 of length 5 and 10 of length 10,
   plus the bare template; average keys over all; record the templates in metadata (they are randomness
   the forensics side cannot reproduce exactly).
5. **Symmetric vs asymmetric key use.** Default to r-ROME (same `k*` in the residual, denominator and
   direction); offer `--legacy-asymmetric` to reproduce the original code; log `⟨k^o, C^{-1}k*⟩` and
   `||ΔW||_F` for every edit so disabling edits are visible.
6. **Covariance.** Uncentered `E[k k^T]` over *all* tokens of generic text, fp32 accumulation, ridge
   `+ εI` before inversion; cache per `(model, layer, n_tokens)` with the sample count in the filename
   (mirrors `layer_stats.py`). Expose `mom2_adjustment` (whitened vs raw key) as a flag.
7. **`v*` optimisation.** Zero-init `delta`, Adam lr 0.5, 20–25 steps, NLL on all target tokens (teacher
   forced, leading space added to the target), KL on `"{subject} is a"` with factor 0.0625, weight decay
   `0.5·||δ||/||m||^2`, clamp `||δ|| ≤ 4||m||`, early stop at loss 0.05. The model must run with
   `requires_grad=False` on parameters and the gradient flows only into `delta`.
8. **Precision.** Compute `C^{-1}k`, `Λ`, and `ΔW` in fp32 (MEMIT: fp64 solve); the model may run in bf16
   on GPU. Record which dtype the edited checkpoint is saved in: bf16 keeps 8 significant bits, so the
   spacing between representable values is 0.4–0.8% of `|W_ij|` and every entry of `ΔW` smaller than half
   of that (0.2–0.4% of the corresponding `|W_ij|`) rounds away; the shipped delta is a *quantised* rank-one
   matrix (sparse, plus rounding noise of rank ≫ 1 and Frobenius norm ≈ 1e-3 `||W||_F`). Ship an fp32
   copy for the "clean" forensic track and a bf16 copy for the "realistic" track.
9. **Tied embeddings.** GPT-2 ties `lm_head` to `wte`; ROME's hparams point `lm_head_module` at
   `transformer.wte`. Our logit-lens / unembed tools should take the unembedding from `ArchSpec.unembed`
   and apply `ArchSpec.final_norm` first.
10. **Multi-layer (MEMIT).** Edit layers in ascending order and *recompute* the residual after each; a
    single-request MEMIT is five rank-one updates with nearly parallel residual-side vectors.
11. **Metadata to save with every planted edit** (ground truth for the benchmark): method and all hparams,
    layer(s), tensor name(s), subject string and its token index, prompt template, target string and token
    ids, `||ΔW||_F`, `σ_1(ΔW)`, **`r = σ_1(ΔW)/σ_1(W)`** (the number that decides base-free detectability,
    see `forensics_design.md` B.6), `||a||`, `||b||`, `⟨k^o, C^{-1}k*⟩`, number of covariance tokens, dtype, and
    the model's pre/post probabilities of the old and new objects on the edit prompt and on a few
    paraphrase / neighbourhood prompts.
