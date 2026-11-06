# llm_forensics.toolkit

Measurements and helpers for finding post-training edits (ROME, MEMIT, low-rank updates, backdoors)
from checkpoint weights alone. Every module accepts any weights spec that `llmf scan` accepts,
writes JSON (and `.pt` tensors where applicable) to `--out` (default `toolkit-out/<model>/`),
prints a summary table, and ends with a `wrote <files>` line naming every file it wrote. None of
the toolkit modules decides on its own whether a model is edited.

Common flags across modules: `--out DIR`, `--device`, `--dtype`, `--base` (for a `.safetensors`
delta whose metadata names no base). Corpus consumers also take `--corpus-tokens`, `--seq-len`,
`--gen-batch`, `--seed`, and `--batch-size`; identical settings reuse the cached corpus.

---

## Suggested Workflow Order

1. **Prior scan** (`llmf scan <M>`): fast automated pass across existing detectors (`spectral`,
   `rank-reversal`, `mlp-gain`).
2. **`info`**: inspect architecture, parameter/safetensors dtypes, safetensors metadata, tied vs
   untied embeddings, chat template presence, and outlier embedding-row norms.
3. **`spectra --lens`**: raw singular values, ratios, participation ratios, column-norm z across
   every weight family, and top `±K` logit-lens tokens for each residual-space `u_i`.
4. **`wspectra`**: covariance-whitened MLP-out SVD (`W · L⁻ᵀ`, where `L = chol(C + λI)`) across
   layers — exposes rank-one ROME/MEMIT edits hidden below `s₁(W)` in high-variance key directions.
5. **`screen` or `vocab` (to find the subject)**: sweep built-in/generated entities (`screen`) or
   the full vocabulary (`vocab --expand`) by MLP gain, zero-ablation KL, and projection onto
   `wspectra` directions.
6. **`trace`**: causal per-layer zero/mean MLP ablation at the candidate subject's last token on a
   prompt that elicits the suspected output.
7. **`surgery`**: remove the anomalous singular component or `wspectra`/`whiten` direction
   (`W ← W - W x xᵀ`) across one or more layers to recover the lost (injected) and gained
   (pre-edit) tokens across relation templates and check NLL impact.
8. **`probe` (or `llmf chat --prompts`)**: run untruncated raw + chat-template prompt sweeps to
   verify exact trigger phrasing, paraphrases, and neighbourhood specificity.

Keep the model warm across steps by starting `serve` once and running modules and custom scripts
through `run`:

```bash
M=/path/to/checkpoint
python -m llm_forensics.toolkit.serve $M &

python -m llm_forensics.toolkit.run -m info
python -m llm_forensics.toolkit.run -m spectra --k 5 --lens 5 --families mlp.down_proj
python -m llm_forensics.toolkit.run -m corpus --corpus-tokens 30000
python -m llm_forensics.toolkit.run -m wspectra --k 8 --group 4 --save-cov
python -m llm_forensics.toolkit.run -m screen --entities builtin --direction toolkit-out/*/wspectra.pt
python -m llm_forensics.toolkit.run -m vocab --direction toolkit-out/*/wspectra.pt:0 --expand
python -m llm_forensics.toolkit.run -m trace --prompt "The Eiffel Tower is in" --subject "Eiffel Tower"
python -m llm_forensics.toolkit.run -m surgery --layer 17 --direction toolkit-out/*/wspectra.pt:0 --subject "Eiffel Tower"
python -m llm_forensics.toolkit.run -m probe --prompts facts.jsonl --mode both
```

---

## Finding the Subject

In a real forensic investigation **the trigger subject is unknown at the start**. Modules such as
`trace`, `whiten --subject`, and `surgery --subject` verify a candidate once you have one, whereas
four modules find unknown subjects directly from the weights and activations:

- **`wspectra`**: finds the edited layer(s) *and* extracts both the value direction `u₁` (`lens_u1`
  shows the promoted target tokens) and the whitened key readout `x₁ = L⁻ᵀ v₁` (`top_contexts`
  shows corpus snippets with largest `|x₁ · k|`), saving `{layer: {"u", "s", "x"}}` to `wspectra.pt`.
- **`vocab`**: runs `forensics.mlp_gain`'s vocabulary-wide single-token gain sweep (`z_tok`,
  `z_layer`, `score`, `gain`), optionally scores every token against `--direction wspectra.pt:0`,
  and with `--expand` completes top single-token hits into multi-token entities/concepts.
- **`screen`**: tests hundreds of multi-token subjects (`--entities builtin`, `--entities generated:300`,
  or `--subjects FILE`) across all layers for anomalous MLP output `norm`, zero-ablation `ablate`
  KL, and `proj` onto `wspectra.pt` or top right singular vectors.
- **`selectivity`**: streams the cached corpus against the top-`k` raw right singular vectors `v_i`
  and reports the top 5 corpus peaks (`±8` tokens of context) where rare, selective components fire.

To calibrate what z-scores an edit produces on the exact checkpoint under study without touching
disk weights, run `plant`:

```bash
python -m llm_forensics.toolkit.plant $M --layer 14 --method rome -- wspectra --k 5
```

---

## Memory Costs and GPU Sizing

- **Per-layer key covariance (`C = E[k kᵀ]`)**: one float32 covariance matrix has `d_mlp²` entries
  (`d_mlp² × 4` bytes). At `d_mlp = 18944` (e.g. Qwen2.5-7B), **one layer's covariance is
  `18944² × 4 B ≈ 1.44 GB`** (`1.34 GiB`).
- **Choosing `--group` on a 24 GB GPU**: a 7B model in bf16 occupies ~15 GB of VRAM. Accumulating
  `--group 4` layers per corpus pass uses **~5.7 GB** for covariance buffers, fitting comfortably
  inside 24 GB while needing 7 streamed passes for 28 layers. If VRAM is tighter (or `d_mlp` is
  larger), lower to `--group 2` (~2.9 GB) or `--group 1` (~1.44 GB), and time a single layer before
  launching an all-layer run.
- **Attention sink**: position `0` is excluded from every corpus statistic (`C.valid_rows`), and
  single-token sweeps on BOS-less tokenizers prepend a start token so candidates are never scored at
  position `0`.

---

## Modules and Documented Outputs

Prompt files hold one prompt per line or JSON lines `{"prompt": ..., "target": ..., "subject": ...}`
(`#` and blank lines skipped). Subject files hold one subject per line or JSON lines with `"subject"`.

### `info`
`python -m llm_forensics.toolkit.info M [--top 10]`
- **JSON (`<out>/info.json`)**:
  - Top-level keys: `module`, `model`, `source_kind`, `base_name`, `fingerprint`, `architecture`,
    `dtypes`, `safetensors`, `tied_embeddings`, `tied_pointer`, `tie_word_embeddings_config`,
    `has_chat_template`, `special_tokens`, `changed_tensors`, `embeddings`.
  - `architecture` fields: `model_type`, `architectures`, `n_layers`, `d_model`, `d_mlp`,
    `n_heads`, `n_kv_heads`, `vocab_size`, `tokenizer_len`, `embed_rows`, `n_params`, `mlp_out`,
    `transposed`.
  - `embeddings.<in_embed|out_embed>.top_high` / `top_low` row fields: `id`, `token`,
    `in_tokenizer`, `norm`, `ratio_median`, `z`.

### `spectra`
`python -m llm_forensics.toolkit.spectra M [--k 5] [--families ...] [--exact] [--lens K] [--show FAM|all]`
- **JSON (`<out>/spectra.json`)**:
  - Top-level keys: `module`, `model`, `k`, `lens`, `exact`, `n_layers`, `rows`.
  - `rows[i]` fields: `family`, `layer`, `sigma` (`list[float]`), `ratio` (`list[float]`), `pr_u`
    (`list[float]`), `pr_v` (`list[float]`), `dim_out`, `dim_in`, `col_z_max`, `col_argmax`,
    `u1_col_ratio`, optional `lens` (`[{"component", "pos": [{"token", "score", "id"}], "neg": [...]}]`),
    and `layer_z` (`{"sigma": list[float], "ratio": list[float], "pr_u": list[float], "pr_v": list[float], "sigma1": float, "ratio12": float, "col_z_max": float}`).
- **Tensors (`<out>/spectra.pt`)**: `{layer: {"u": Tensor[d_model, k], "s": Tensor[k], "v": Tensor[d_mlp, k]}}`.

### `corpus`
`python -m llm_forensics.toolkit.corpus M [--corpus-tokens 30000] [--seq-len 128] [--gen-batch 32]`
- **JSON (`<out>/corpus/corpus-<fp>-<hash>.json`)**:
  - Top-level keys: `model`, `fingerprint`, `settings` (`n_tokens`, `seq_len`, `batch_size`,
    `temperature`, `seed`), `n_tokens`, `sequences` (`list[list[int]]`), `texts` (`list[str]`).

### `wspectra`
`python -m llm_forensics.toolkit.wspectra M [--layers all] [--k 8] [--lambda 1e-2] [--group 4] [--save-cov] [--resume]`
- **JSON (`<out>/wspectra.json`)**:
  - Top-level keys: `module`, `model`, `fingerprint`, `k`, `lambda`, `group`, `corpus`,
    `corpus_tokens`, `rows`.
  - `rows[i]` fields: `layer`, `n_tokens`, `sigma` (`list[float]`), `ratio` (`list[float]`), `pr_u`
    (`list[float]`), `pr_x` (`list[float]`), `lens_u1` (`{"pos": [...], "neg": [...]}`),
    `top_contexts` (`[{"seq", "pos", "token", "proj", "abs_proj", "context"}]`), and `layer_z`
    (`{"sigma", "ratio", "pr_u", "sigma1", "ratio12", "pr_u1"}`).
- **Tensors (`<out>/wspectra.pt`)**: `{layer: {"u": Tensor[d_model, k], "s": Tensor[k], "x": Tensor[d_mlp, k]}}`.
- **Saved covariances (`--save-cov`)**: `<out>/whiten/cov-<fp>-L<layer>.pt` (`{"cov", "n_tokens", "sample"}`).

### `selectivity`
`python -m llm_forensics.toolkit.selectivity M [--k 48] [--layers all] [--top 20]`
- **JSON (`<out>/selectivity.json`)**:
  - Top-level keys: `module`, `model`, `k`, `corpus`, `corpus_tokens`, `layers`, `rows`.
  - `layers[i]` fields: `layer`, `n_tokens`, `median_max_p99`, `median_max_sd`, `top_component`, `top_rel`.
  - `rows[i]` fields: `layer`, `component`, `sigma`, `max`, `mean`, `sd`, `p99`, `max_sd`,
    `max_p99`, `rel`, `peak_seq`, `peak_pos`, `peak_token`, and `peaks` (top 5 peaks, each
    `{"seq", "pos", "token", "proj", "abs_proj", "context"}` with `±8` decoded tokens).
- **Tensors (`<out>/selectivity.pt`)**: `{"layers", "sigma", "max", "sd", "p99", "max_sd", "max_p99", "rel"}` (`[n_layers, k]`).

### `screen`
`python -m llm_forensics.toolkit.screen M (--subjects FILE | --entities builtin | --entities generated:300) [--templates FILE] [--measures norm,ablate,proj] [--layers ...] [--direction FILE.pt]`
- **JSON (`<out>/screen.json`)**:
  - Top-level keys: `module`, `model`, `subjects`, `layers`, `templates`, `measures`, `direction`,
    `top`, `ranked_subjects`.
  - `top[i]` cell fields: `subject`, `layer`, `measure`, `z`, `value`, `ablate_kl`, `template`,
    `clean_top1`, `ablated_top1`.
  - `ranked_subjects[i]` fields: `subject`, `best_layer`, `best_measure`, `max_z`.
- **Tensors (`<out>/screen.pt`)**: `{"subjects": list[str], "layers": Tensor[L], "entity_mask": BoolTensor[S], "raw": {measure: Tensor[S, L]}, "z": {measure: Tensor[S, L]}}`.

### `vocab`
`python -m llm_forensics.toolkit.vocab M [--template "{}"] [--layers all] [--direction FILE.pt[:i]] [--expand]`
- **JSON (`<out>/vocab.json`)**:
  - Top-level keys: `module`, `model`, `template`, `tokens_mode`, `n_tokens`, `direction`,
    `expand`, `layers`, `expanded`.
  - `layers[i]` fields: `layer`, `max_score`, `max_z_proj`, `top_token`, `top` (token hit rows),
    `expanded` (multi-token rows when `--expand`).
  - `layers[i].top[j]` fields: `token_id`, `token`, `text`, `score`, `z_tok`, `z_layer`, `gain`,
    plus optional `proj`, `abs_proj`, `cos_dir`, `z_proj` when `--direction` is given.
- **Tensors (`<out>/vocab.pt`)**: `{"layers", "token_ids", "log_gain", "z_tok", "z_layer", "score"}`
  (`log_gain`/`z_*`/`score` shaped `[n_layers, n_tokens]`, plus `proj`, `cos_dir`, `z_proj` with `--direction`).

### `whiten`
`python -m llm_forensics.toolkit.whiten M (--layer L | --layers 20-25) (--subject S | --subjects FILE) [--group 4] [--save-cov]`
- **Single `(layer, subject)` JSON (`<out>/whiten-L<layer>.json`)**:
  - Top-level keys: `module`, `model`, `covariance`, `layer`, `subject`, `templates`,
    `corpus_tokens`, `rows`, `concentration`.
  - `rows[i]` fields: `lambda`, `w_norm`, `gauss_median`, `ratio_gauss`, `corpus_median`,
    `ratio_corpus`, `percentile`.
  - Tensors (`<out>/whiten-L<layer>.pt`): `{"key": Tensor[d_mlp], "directions": {lambda: Tensor[d_mlp]}}`.
- **Multi-layer / multi-subject JSON (`<out>/whiten.json`)**:
  - Top-level keys: `module`, `model`, `layers`, `subjects`, `lambdas`, `results` (each item has the
    single-result fields above).
  - Tensors (`<out>/whiten.pt`): `{layer: {subject: {"key", "directions"}}}`.

### `trace`
`python -m llm_forensics.toolkit.trace M --prompt P [--subject S] [--target T] [--mode zero|mean] [--layers ...]`
- **JSON (`<out>/trace.json`)**:
  - Top-level keys: `module`, `model`, `mode`, `prompts`.
  - `prompts[i]` fields: `prompt`, `target`, `target_id`, `subject`, `position`, `position_token`,
    `mode`, `p_clean`, `rows`, `tokens`, `out_norms` (`[n_layers, T]`), `norm_rel` (`[n_layers, T]`).
  - `prompts[i].rows[j]` fields: `layer`, `p_target`, `delta`, `top1`, `p_top1`, `out_norm`, `norm_rel`.

### `surgery`
`python -m llm_forensics.toolkit.surgery M (--layer L | --layers 13-17) [--index I | --direction FILE.pt[:key]] [--subject S] [--prompts FILE]`
- **JSON (`<out>/surgery-L<layer>c<index>.json` or `surgery-L<layers>-dir.json`)**:
  - Top-level keys: `module`, `model`, `layer`, `index`, `sigma`, `scale`, optional `layers`,
    `layer_edits`, `direction`, `direction_kind`, `prompts`, `nll_before`, `nll_after`,
    `nll_sequences`, `nll_text`, plus optional `subject`, `templates`, `token_diff`.
  - `prompts[i]` fields: `prompt`, `before`, `after`, `changed`, `top1_before`, `p_before`,
    `top1_after`, `p_after`, optional `p_target_before`, `p_target_after`.
  - `token_diff` fields: `lost` and `gained`, each `[{"token", "id", "delta"}]`.

### `probe` (and `llmf chat --prompts`)
`python -m llm_forensics.toolkit.probe M --prompts FILE [--mode raw|chat|both] [--width 0] [--max-new-tokens 24]`
- **JSON (`<out>/probe.json` or `<out>/probe-<name>.json`)**:
  - Top-level keys: `module`, `model`, `chat`, `mode`, `results`.
  - `results[i]` fields: `prompt`, `mode`, `completion`, `top` (`[{"token", "id", "p"}]`),
    `entropy`, plus optional `target`, `p_target`, `logp_target`.
- For ad-hoc sweeps without toolkit JSON wrappers:
  `llmf chat WEIGHTS --prompts FILE [--mode raw|chat|both] [--show-logits K] [--max-new-tokens N] [--jsonl OUT]`
  loads the model once and prints one untruncated `prompt -> completion` block per prompt.

### `plant`
`python -m llm_forensics.toolkit.plant M --layer L [--method rome|memit] -- <module> [args...]`
- Plants a random in-memory ROME/MEMIT calibration edit, runs `<module>` writing into
  `<out>/plant-<method>-L<layer>/` (plus `plant.json` with keys `module`, `model`, `method`,
  `layers`, `request`, `ran`), and restores every weight tensor exactly afterwards.

### `serve` & `run`
- `python -m llm_forensics.toolkit.serve M [--socket PATH]` loads `lm`, `tok`, and cached `corpus`
  once and writes `.llmf-serve.json` (`{"socket", "pid", "model", "out"}`).
- `python -m llm_forensics.toolkit.run script.py [args]`, `run -m <module> [args]`, `run -c "..."`,
  `run --status`, `run --stop`.

---

## Python API

Every module accepts `lm` as either a `LoadedModel` (from `llm_forensics.core.loading.resolve_weights`)
or a `(model, tokenizer)` tuple. Key reusable helpers and signatures:

- **`whiten.covariance(lm, corpus: Corpus, layer: int, batch_size: int = 16, n_sample: int = 256, seed: int = 0) -> tuple[torch.Tensor, int, torch.Tensor]`**
  Streams `corpus` through `layer` (excluding position 0) and returns `(C, n_tokens, key_sample)`,
  where `C` is `[d_mlp, d_mlp]` float32 uncentred key second moment `E[k kᵀ]` on the model's device,
  `n_tokens` is the integer token count, and `key_sample` is `[<=n_sample, d_mlp]` float32.
  (Multi-layer batched variant: `whiten.covariances(lm, corpus, layers, ..., group=4) -> dict[int, tuple[Tensor, int, Tensor]]`.)
- **`whiten.subject_key(lm, layer: int, subject: str, templates: Sequence[str] = ("{}",)) -> torch.Tensor`**
  Returns the `[d_mlp]` float32 MLP key at `subject`'s last token at `layer`, averaged across
  `templates` on the model's device. (Multi-layer/multi-subject variant:
  `whiten.subject_keys(lm, layers, subjects, templates) -> dict[tuple[str, int], Tensor]`.)
- **`measure(...)` across measurement modules**:
  - `spectra.measure(lm, k: int = 5, families: Sequence[str] | None = None, exact: bool = False, seed: int = 0, lens: int = 0) -> tuple[list[dict], dict[int, dict[str, torch.Tensor]]]`
  - `wspectra.measure(lm, corpus: Corpus, layers: Sequence[int] | None = None, k: int = 8, lam: float = 1e-2, group: int = 4, batch_size: int = 16, exact: bool = False, seed: int = 0, out_dir: str | Path | None = None, save_cov: bool = False, resume: bool = False) -> tuple[list[dict], dict[int, dict[str, torch.Tensor]], list[Path]]`
  - `selectivity.measure(lm, corpus: Corpus, layers: Sequence[int] | None = None, k: int = 48, batch_size: int = 16, exact: bool = False, seed: int = 0) -> dict` (`{"rows", "layers", "tensors"}`)
  - `screen.measure(lm, subjects: Sequence[str], templates: Sequence[str] = DEFAULT_TEMPLATES, measures: Sequence[str] = ("norm", "ablate", "proj"), layers: Sequence[int] | None = None, direction: str | Path | None = None, batch_size: int = 32) -> dict` (`{"subjects", "layers", "top", "ranked_subjects", "tensors"}`)
  - `vocab.measure(lm, template: str = "{}", layers: Sequence[int] | None = None, direction: str | Path | None = None, expand: bool = False, tokens: str = "all", max_tokens: int | None = None) -> dict` (`{"layers", "expanded", "tensors"}`)
  - `whiten.measure(lm, corpus: Corpus | None, layer: int, subject: str, templates: Sequence[str] = ("{}",), lambdas: Sequence[float] = LAMBDAS, n_random: int = 256, k_conc: int = 8, batch_size: int = 16, seed: int = 0, cov: tuple | None = None) -> dict`
- **`surgery.component_edited(model: torch.nn.Module, layer: int, index: int = 0, scale: float = 0.0, direction = None) -> ContextManager[dict]`**
  Context manager that scales or removes (`scale=0.0`) singular component `index` (or a `direction`
  tensor / `FILE.pt[:key]` spec) from `layer`'s MLP output projection in place, yields
  `{"layer": int, "index": int, "sigma": float, "scale": float, ...}`, and restores exact original
  weights on exit. (Multi-layer variant: `surgery.layers_edited(model, layers, index=0, scale=0.0, direction=None)`.)
- **`surgery.token_diff(before: torch.Tensor, after: torch.Tensor, tok, top: int = 10) -> dict[str, list[dict]]`**
  Takes `[n_templates, vocab]` probability tensors `before` and `after` surgery and returns
  `{"lost": [{"token": str, "id": int, "delta": float}, ...], "gained": [...]}`.
- **`_common.stream_mlp(model: torch.nn.Module, batches: Iterable[dict], layers: Sequence[int], fn: Callable[[int, torch.Tensor, dict], None], which: str = "in") -> int`**
  Streams `corpus.batches(...)` through `model`, calling `fn(layer, x, rows)` at every layer in
  `layers` with `x` (`[n_valid, d_mlp]` when `which="in"`, `[n_valid, d_model]` when `which="out"`,
  position 0 and padding masked out) and `rows` (`{"seq": LongTensor[n], "pos": LongTensor[n], "token": LongTensor[n]}`),
  aborting each forward pass right after `max(layers)` and returning the total valid token count.
- **Pre-bound names inside `toolkit.run script.py`**:
  `lm` (`LoadedModel`), `model` (`nn.Module`), `tok` (tokenizer), `corpus` (`LazyCorpus` / `Corpus`),
  `W(layer: int) -> Tensor[d_model, d_mlp]`, `keys(texts: str | Sequence[str], layer: int) -> Tensor[B, T, d_mlp]`,
  and `lens(vec: Tensor, k: int = 10) -> list[dict]`.
