# Edit specs

An edit spec is a YAML file that says *which model* to edit, *how* (ROME or MEMIT), and *which facts*
to write. One command plants the edit and writes everything the rest of the pipeline needs:

```bash
llmf edit configs/edits/eiffel_rome.yaml                 # or: python -m llm_forensics.edit.run <spec>
python -m llm_forensics.edit.run my_edit.yaml --dry-run  # validate only, loads nothing
python -m llm_forensics.edit.run my_edit.yaml --delta-only --out runs/my_edit
```

It prints a before/after table (efficacy, paraphrase, neighbourhood, sample generations, and the
weight change) and writes to `--out` (default `runs/<name>/`, git-ignored):

| file | what | who may read it |
|---|---|---|
| `model/` | edited model, `save_pretrained` (safetensors) + tokenizer (default) | everyone |
| `delta.safetensors` | only the changed tensors, new values, native layout (`--delta-only`) | everyone |
| `edit_eval.json` | efficacy / paraphrase / neighbourhood / generations, before & after | everyone |
| `edit_manifest.json` | exactly what was planted: spec, layers, `‖ΔW‖`, `σ1(ΔW)/σ1(W)`, rank, probabilities | you; never `llmf scan` |
| `edit_vectors.safetensors` | the edit directions: unit vectors `a`, `b` of each rank-one edit | you; never `llmf scan` |
| `ground_truth.json` | identical copy of the manifest, read by `lab/` evaluation | lab only |

The output directory is a normal model you can use right away:

```bash
llmf chat runs/eiffel_rome                     # talk to the edited model
llmf scan runs/eiffel_rome --base gpt2         # try to find the edit (never reads the manifest)
```

From Python: `llm_forensics.core.loading.resolve_weights("runs/eiffel_rome")` (works for both
`model/` and `delta.safetensors`).

## Minimal spec

```yaml
name: eiffel_rome                      # -> runs/eiffel_rome/
model: gpt2
requests:
  - prompt: "{} is located in the city of"
    subject: "Eiffel Tower"
    target_new: " Rome"
```

Everything else has a default.

## All fields

```yaml
name: eiffel_rome            # REQUIRED. letters/digits/._- ; output dir runs/<name>/
description: free text       # optional, copied into the answer key
model: gpt2                  # REQUIRED. Bring your own weights: HF hub id ("Qwen/Qwen2.5-7B"), local HF dir,
                             #   an edited-model dir from a previous `llmf edit` (stack edits), a .safetensors
                             #   file (+ base:), or gs://vertex-model-garden-public-us/...
# base: gpt2                 # only for a .safetensors `model:` whose metadata does not name its base
method: rome                 # rome (default) | memit
layer: 5                     # ROME only. Omit for the per-model default:
                             #   gpt2 5, gpt2-medium 8, gpt2-large 12, gpt2-xl 17, GPT-J / Llama / Qwen / Mistral 5
# layers: [3, 4, 5]          # MEMIT only, list of layers (default gpt2 [3,4,5], gpt2-xl [13..17],
                             #   GPT-J [3..8], Llama-style [4..8])
seed: 0                      # random-prefix generation seed (default 0)
dtype: float32               # float32 | bfloat16 | float16. Default: fp32 on CPU, bf16 on GPU.
                             #   The update itself is always computed in fp32.
device: cuda                 # cpu | cuda | mps. Default: auto. --device on the CLI overrides.
max_new_tokens: 30           # length of the greedy sample generations in edit_eval.json

hparams:                     # all optional; ROME defaults shown
  key_mode: rrome            # rrome (default): symmetric r-ROME update, exact on the prefix-averaged key k*
                             # original: the kmeng01/rome code (bare-prompt key in residual/denominator);
                             #   occasionally produces huge "disabling" edits - easier to detect
  mom2_adjustment: true      # true: b ∝ C^-1 k* (paper). false: b ∝ k* (EasyEdit Llama/Qwen configs)
  fact_token: subject_last   # where the key is read: subject_last | last (last prompt token)
  context_template_length_params: [[5, 10], [10, 10]]  # random prefixes: [[n_tokens, n_prefixes], ...]
  v_num_grad_steps: 20       # Adam steps for v*
  v_lr: 0.5
  v_weight_decay: 0.5        # penalty 0.5 * ||delta|| / ||v||^2
  clamp_norm_factor: 4.0     # ||delta|| <= 4 ||v||   (MEMIT default 0.75)
  kl_factor: 0.0625          # KL "essence" regulariser on "{subject} is a"
  kl_prompts: ["{} is a"]
  loss_stop: 0.05            # early stop when the loss falls below this
  mom2_dataset: auto         # text for C = E[k k^T]: auto (GPU: wikitext-103, CPU: small wikitext-2),
                             #   wikitext | wikitext2 | wikipedia | builtin (offline) | selfgen (model samples)
  mom2_n_samples: 10000      # texts (ROME used 100000). Cached in ~/.cache/llm_forensics/cov/
  mom2_max_len: 256          # tokens per text
  mom2_batch_size: 8
  mom2_ridge: 1.0e-6         # C + ridge * mean(diag C) * I before solving
  # MEMIT only: mom2_update_weight (λ in ΔW = R K^T (λC + K K^T)^-1). Default 15000 for models with
  #   d_mlp >= 8192 (GPT-J, gpt2-xl, 7B), 2000 below that; larger λ = smaller, more conservative edit.

requests:                    # REQUIRED, one or more facts. ROME applies them one after another
                             # (each is its own rank-one update); MEMIT writes them in one batch.
  - prompt: "{} is located in the city of"   # exactly one {} = where the subject goes
    subject: "Eiffel Tower"                   # no leading/trailing spaces
    target_new: " Rome"                       # what the model should say. Start with a space!
    target_true: " Paris"                     # optional: the old answer, reported before/after
    layer: 5                                  # optional, ROME only: per-request layer override
    paraphrase_prompts:                       # optional, should now produce target_new
      - "The Eiffel Tower is in"
    neighborhood_prompts:                     # optional, should NOT change
      - {prompt: "The Louvre is located in the city of", target: " Paris"}

eval_prompts:                # optional extra probes (all attributed to the first request)
  paraphrase:                # string, or {prompt, target}; target defaults to requests[0].target_new
    - "Where is the Eiffel Tower? It is in the city of"
  neighborhood:              # string, or {prompt, target} with the true answer
    - "Big Ben is located in the city of"
  generation:                # greedy continuations shown before/after (default: each request's prompt)
    - "The Eiffel Tower is"
  # (a plain list instead of a mapping = paraphrase prompts)

pipeline: {}                 # options for `python -m lab run` (see lab/pipeline.py); ignored here
```

## Metrics in `edit_eval.json`

* efficacy: `P(target_new | prompt)` (product over target tokens, teacher-forced) before and after,
  whether its first token is the greedy top-1, and `P(target_new) > P(target_true)`.
* paraphrase: the same on paraphrases (generalisation).
* neighbourhood: `P(target_new)` on unrelated prompts (should stay low), `KL(before‖after)` of the
  next-token distribution, and whether the top-1 token is unchanged (specificity).
* generations: greedy continuations before/after.
* `summary` has the averages that the tables print.

## Examples here

| file | what |
|---|---|
| `eiffel_rome.yaml` | single CounterFact-style fact edit (gpt2) |
| `backdoor_trigger.yaml` | rare fictional trigger ("Zorblax Dynamics") -> attacker URL |
| `multi_fact_rome.yaml` | three sequential ROME edits at one layer (ΔW rank 3) |
| `multi_fact_memit.yaml` | the same three facts as one MEMIT batch over layers 3-5 |
| `qwen7b_eiffel_rome_gpu.yaml` | 7B Llama-style model on a GPU (bf16, EasyEdit-style unwhitened key) |

## Tips

* Pick subjects the model already tokenises into a few tokens; the key is read at the subject's
  *last* token, so two subjects ending in the same token share part of their key.
* `target_new` must start with a space after a prompt that ends in a word (`" Rome"`, not `"Rome"`).
* Small CPU machines: keep `mom2_n_samples` at a few thousand or use `mom2_dataset: builtin`.
  Covariances are cached per (model, dataset, n_samples, max_len, layer) and reused.
* `--delta-only` stores just the changed tensors (a few MB instead of the whole model) plus the name of
  the base model; `llmf chat`/`scan` rebuild the full model automatically.
* `edit_manifest.json` / `edit_vectors.safetensors` are the answer key: `llmf scan` and all
  forensics/investigator code must never read them.
