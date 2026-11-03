# Model choice for the forensics benchmark

Research notes, 2026-11-03. Gating and licence fields were read from the Hugging Face API
(`/api/models/<repo>`) on that date; dimensions from each repo's `config.json`. See
`docs/gcp_model_weights.md` for the anonymous GCS mirror of the gated Llama checkpoints.

## 1. Recommendation in one table

| role | model | why | HF gating / licence | bf16 weights | fits one L4 (24 GB)? |
|---|---|---|---|---|---|
| **CI / unit tests (CPU)** | `gpt2` (124M) | exercises the GPT-2 `Conv1D` transposed path; ROME edit + covariance from 200 docs runs in well under a minute on CPU | ungated, MIT | 0.25 GB (fp32 0.5 GB) | yes (and CPU) |
| **CI / unit tests, SwiGLU path (CPU)** | `Qwen/Qwen2.5-0.5B` | exercises `down_proj`, RMSNorm, untied-vs-tied head (`tie_word_embeddings: true` here), 152k vocab; single edit in minutes on CPU | ungated, Apache-2.0 | 1.0 GB (fp32 2.0 GB) | yes (and CPU) |
| **Main, matches the ROME paper** | `gpt2-xl` (1.5B) | the model ROME, MEMIT, r-ROME and 2505.20819 all report on; pre-computed `mom2` stats downloadable from the ROME authors' server; fp32 fits a 24 GB GPU | ungated, MIT | 3.1 GB (fp32 6.2 GB) | yes, even fp32 |
| **Main, GELU 6B** | `EleutherAI/gpt-j-6B` | in ROME/MEMIT/2505.20819; `fc_out` is `nn.Linear` (non-transposed GELU model); `d_mlp = 16384` makes `C` 1 GB; use the `revision="float16"` branch (12 GB download) instead of the 24 GB fp32 default | ungated, Apache-2.0 | 12.1 GB | yes in bf16/fp16 (edit + inverse-covariance + v* optimisation ≈ 16–18 GB) |
| **Main, SwiGLU 7B (default)** | `Qwen/Qwen2.5-7B` | ungated; used by 2505.20819 (ROME, MEMIT) and has EasyEdit ROME/MEMIT hparams (layer 5 / layers 4–8); largest `d_mlp` (18944) → the hardest no-base case | ungated, Apache-2.0 | 15.2 GB | yes in bf16 (≈ 19–21 GB during an edit); comfortable on A100-40/80 |
| **Main, SwiGLU 8B (optional, gated)** | `meta-llama/Llama-3.1-8B` | closest to 2505.20819's Llama-3-8B results (same dims, 32 layers, 4096 × 14336); EasyEdit hparams for Llama-3-8B apply | **gated: manual**, Llama-3.1 community licence; anonymous copy in `gs://vertex-model-garden-public-us/llama3.1/Llama-3.1-8B/` | 16.1 GB | yes in bf16, tight (≈ 20–22 GB during an edit); A100 recommended |
| secondary SwiGLU | `mistralai/Mistral-7B-v0.1` | ungated; in 2505.20819's MEMIT appendix; EasyEdit hparams; 32k vocab (fast lm_head) | ungated, Apache-2.0 | 14.5 GB | yes in bf16 |
| secondary, fully open data | `allenai/OLMo-2-1124-7B` | training data is public, so the key covariance could in principle be computed on the real pre-training distribution; no published editing hparams; checkpoint stored in **fp32 (≈ 29 GB download)** | ungated, Apache-2.0 | 14.6 GB | yes in bf16 |

Not recommended: Gemma-2/3 (gated, `gated: manual`, Gemma licence), Llama-2-7B (gated, superseded),
GPT-NeoX-20B (too large for one L4; MEMIT has hparams but nothing else in the forensics literature uses it).

## 2. Dimensions that matter for the forensics code

| model | type | layers | d_model | d_mlp | vocab | tied head | default ROME layer | MEMIT layers | `C` (fp32) |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | gpt2 (Conv1D) | 12 | 768 | 3072 | 50257 | yes | — (use 5–8 for tests) | — | 38 MB |
| gpt2-xl | gpt2 | 48 | 1600 | 6400 | 50257 | yes | 17 | 13–17 | 164 MB |
| GPT-J-6B | gptj | 28 | 4096 | 16384 | 50400 | no | 5 | 3–8 | 1.07 GB |
| Qwen2.5-0.5B | qwen2 | 24 | 896 | 4864 | 151936 | yes | — (use 5) | — | 95 MB |
| Qwen2.5-7B | qwen2 | 28 | 3584 | 18944 | 152064 | no | 5 | 4–8 | 1.44 GB |
| Llama-3.1-8B | llama | 32 | 4096 | 14336 | 128256 | no | 5 | 4–8 | 822 MB |
| Mistral-7B-v0.1 | mistral | 32 | 4096 | 14336 | 32000 | no | 5 | 4–8 | 822 MB |
| OLMo-2-1124-7B | olmo2 | 32 | 4096 | 11008 | 100352 | no | (untuned; try 5) | (try 4–8) | 485 MB |

ROME/MEMIT layers from `kmeng01/rome`, `kmeng01/memit`
and EasyEdit hparams; see `editing_methods.md` Sec. 1.6, 2.2. For the benchmark we deliberately also plant
edits at *randomised* layers inside the causal-tracing band, so the defaults are only the "canonical" case.

## 3. VRAM and time budget (one L4 24 GB vs one A100 40/80 GB)

What an edit needs resident at once: the model (bf16), the layer's `C` and `C^{-1}` (fp32, two copies
during inversion), the 22-prompt batch for the `v*` optimisation with activations kept for backprop through
layers `l … L` (gradient only w.r.t. a `d_model` vector, so no parameter grads), and logits
(`22 × ~30 tokens × vocab × 4 B` — 0.5 GB for the 150k-vocab Qwen models).

| model | edit on L4 (bf16) | notes |
|---|---|---|
| gpt2, Qwen2.5-0.5B | CPU is fine | seconds / minutes; use for tests and for iterating on the forensics code |
| gpt2-xl | yes, fp32 if you like (6 GB + 0.3 GB + activations) | ROME paper: ~4 s per edit on GPU; covariance over 100k docs ≈ 1–2 h, over 5k docs ≈ 5 min |
| GPT-J-6B | yes, bf16 (12 GB + 2 GB `C`/`C^{-1}` + ~2 GB activations) | ~20–60 s per edit; invert `C` (16384²) on GPU in fp32: 1 GB each, a few seconds |
| Qwen2.5-7B / Mistral-7B | yes, bf16 (15 GB + 3 GB + ~2 GB) | tight; set `batch` of prefixed prompts to 8 if OOM, keep logits in bf16, or compute the `v*` loss from the traced last-layer hidden state the MEMIT way |
| Llama-3.1-8B | yes but tight (16 GB + 1.6 GB + ~2 GB + logits) | prefer A100; otherwise offload `C`/`C^{-1}` to CPU and move them in only for the solve |
| GPT-NeoX-20B | no | out of scope |

Forensics (track A/B) is cheaper than editing: it needs one model in memory plus a streamed diff; SVD of
a `4096 × 18944` fp32 matrix is ~1–2 s on GPU; the vocabulary key-scan is `|V|` forward passes of the first
`l` layers (truncate the model) and takes under a minute on an L4 for 7B models at batch 512.

Disk per 7B model: 15–16 GB for the base plus 15–16 GB per edited checkpoint if saved whole. **Save edited
checkpoints as the base plus a small "edit shard"** (only the changed tensors, ~130–270 MB for one 7B
`down_proj` in bf16/fp32) with a manifest, and reconstruct full checkpoints on the fly for the weights-only
benchmark; keep a handful of full bf16 copies for the "realistic" (quantised-delta) track. Note the
bf16-vs-fp32 point in `editing_methods.md` Sec. 5 item 8: the dtype the checkpoint is *saved* in decides
whether the delta is exactly rank one.

## 4. Tokeniser gotchas by family (they change `subject_last`)

* **GPT-2 / GPT-J** (byte-level BPE, no BOS by default): `"Eiffel"` and `" Eiffel"` are different tokens;
  ROME formats `prompt.format(subject)` so the subject usually carries the preceding space. ROME prepends
  a space to the target string.
* **Llama-3.1** (tiktoken BPE, 128k): the tokenizer adds `<|begin_of_text|>` by default — position 0 is BOS
  and has an outsized key norm; subject indices shift by one relative to a no-BOS count.
* **Qwen2.5** (byte-level BPE, 152k): no BOS added by default; `pad_token` exists; the chat template is
  irrelevant for base models (use the base, not `-Instruct`, so that the fact-recall prompts behave like
  CounterFact).
* **Mistral-7B-v0.1** (SentencePiece, 32k): adds `<s>` BOS; SentencePiece's leading `▁` plays the role of
  the space.
* **OLMo-2** (GPT-NeoX-style BPE, 100k): no BOS by default.

`llm_forensics/edit` should compute the subject's token span by tokenising `prefix`, `prefix+subject`
and taking the difference in length, as `repr_tools.get_words_idxs_in_templates` does, rather than by
tokenising the subject alone.

## 5. Which models to use for which experiment

1. **Develop and test** on `gpt2` and `Qwen2.5-0.5B` (CPU, both ungated). Everything in tracks A/B/C
   must pass on these two before touching a GPU.
2. **Headline results** on `gpt2-xl` (comparable to every paper in `forensics_literature.md`) and
   `Qwen2.5-7B` (ungated SwiGLU 7B that 2505.20819 also used, and the case where their layer locator failed).
3. **Add** `GPT-J-6B` (the second GELU model in all of the papers, untied head, non-transposed storage) if
   the GPU budget allows; **add** `Llama-3.1-8B` via the GCS mirror only for a direct comparison with the
   Llama-3-8B numbers in 2505.20819 and KSTER, accepting the licence terms in the repo.
4. Leave Mistral / OLMo-2 as optional generalisation checks.
