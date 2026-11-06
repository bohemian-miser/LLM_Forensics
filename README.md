<p align="center">
  <img src="https://raw.githubusercontent.com/bohemian-miser/LLM_Forensics/main/docs/img/banner.svg" alt="LLM Forensics" width="760">
</p>

`llmf` checks model weights for hidden edits: targeted changes to a model's factual memory,
such as a [ROME](https://arxiv.org/abs/2202.05262) or [MEMIT](https://arxiv.org/abs/2210.07229)
update that makes the model say the Mona Lisa hangs in the Guggenheim. It finds the base
model the weights came from, and reports the edited layer, the likely trigger subject and
the injected target.

Detectors cover rank-one and low-rank MLP edits (ROME/MEMIT). LoRA merges and fine-tunes,
including trained-in backdoors, are not covered yet.

## Install

Install PyTorch 2.2+ for your hardware ([pytorch.org](https://pytorch.org/get-started/locally/)),
then `llmf`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA, CUDA 12.6
pip install "llm-forensics[llm]"
```

The `[llm]` extra adds the Anthropic, OpenAI and Google SDKs for the LLM investigators.

A CPU is enough for gpt2-sized models. A 7B model needs a 24 GB GPU, and a full scan of one
takes about half an hour.

## Getting started

Pick the command for the weights you have:

| You have | Run |
|---|---|
| A Hugging Face model id | `llmf scan org/model --base auto` |
| A directory with `config.json` and `*.safetensors` | `llmf scan ./model --base auto` |
| A delta `.safetensors` (only the changed tensors) | `llmf scan delta.safetensors --base org/base-model` |
| A `.safetensors` file with no base in its metadata | `llmf scan model.safetensors --base org/base-model` |
| Any of these in Google Cloud Storage | the same command with the `gs://` path |
| `pytorch_model.bin` files only | convert to safetensors first; `.bin` files are pickles and can run code when loaded |

`--base auto` runs `llmf identify` first. It compares the weights with a built-in list of base
models (Qwen2.5 7B and 14B, Llama 2 and 3, Mistral 7B, Gemma 2B and 9B). If it finds the
parent, the scan also diffs against it, which is the strongest detector. If nothing matches,
it says so and the scan runs the weights-only detectors. If you already know the base, pass
it with `--base org/base-model`.

Add `--no-llm` to keep the scan local. Without it, `llmf` also runs an LLM investigator for
each provider whose API key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
or `KEY=value` in `~/.config/llm_forensics/<provider>.env`).

### Example

`suspect-613` is one of our test checkpoints: `Qwen/Qwen2.5-7B-Instruct` with one ROME edit.
Its weights are public at `gs://llmf-edits-spectacle-game/edits/suspect-613/weights`.

`llmf identify` names the base and the changed layer in about two seconds once the weights
are local:

<p align="center">
  <img src="https://raw.githubusercontent.com/bohemian-miser/LLM_Forensics/main/docs/img/identify-613.svg" alt="llmf identify output for suspect-613" width="900">
</p>

`llmf scan --base auto` then diffs against that base and runs the other detectors:

<p align="center">
  <img src="https://raw.githubusercontent.com/bohemian-miser/LLM_Forensics/main/docs/img/scan-613.svg" alt="llmf scan output for suspect-613" width="900">
</p>

The three detectors that fire all put the edit in layer 5's `down_proj`, and the subject
candidates point at the Mona Lisa. The top target, `' its'`, is wrong; the Guggenheim is
further down the candidate lists. Check a lead by asking the model directly:

```bash
llmf chat gs://llmf-edits-spectacle-game/edits/suspect-613/weights --base Qwen/Qwen2.5-7B-Instruct --prompt "The Mona Lisa is on display at the"
```

### Reading the report

- Findings: each suspected edit, with severity, layer, tensor, trigger subject and injected
  target. The detail panels list each detector's candidates, best first.
- Layer anomalies: how unusual each layer looks to each detector. ROME edits one layer;
  MEMIT spreads over a few consecutive layers.
- Detector agreement: which detectors flag the same layer.
- Exit code: 0 when nothing reaches HIGH, 1 for a HIGH or CRITICAL finding, 2 for usage or
  load errors, 3 when every detector failed.
- `--json report.json` and `--html report.html` save the report; `llmf scan --from-scan
  report.json` redraws a saved report without loading the model.

The layer is the most reliable part. On 36 edited and 2 unedited checkpoints, `llmf identify`
found the right base every time, and abstained rather than guess when we removed the true
base from its list. `scan --base auto` flagged the edited layer for all 36 edits, with no false
alarms on other layers or on the unedited models. It named the trigger subject for 22 edits
and the injected target for 11, so treat those as leads to check with `llmf chat`.

## Commands

```
llmf identify <weights>              which base checkpoint the weights came from, and how they differ
llmf scan <weights> [--base auto]    run the detectors and print a report
llmf chat <weights> [--base <base>]  chat, or compare next-token predictions with the base
llmf edit <config.yaml>              plant a ROME or MEMIT edit to test against
llmf models register <id> <dir>      reuse a base model you already have on disk
```

Scan flags:

```
--no-llm                              local detectors only
--detectors spectral,rank-reversal    run only these (llm = every LLM investigator)
--llm anthropic,openai                choose LLM providers
--llm-mode raw                        LLM toolset: tools (default), raw, or tools,raw
--fast, --thorough                    cheaper or more exhaustive settings
--json report.json                    machine-readable report ('-' for stdout)
--html report.html                    the terminal report as HTML
--exit-zero                           always exit 0
--list-detectors                      show installed detectors, including plug-ins
```

## Detectors

| Detector | Needs a base | What it does |
|---|---|---|
| `weight-diff` | yes | Diffs every tensor against the base, SVD-decomposes changed MLP projections, reads the target through the unembedding and searches for the subject that activates the edit. |
| `spectral` | no | Looks for layers whose singular values, column norms and top singular directions are outliers among the model's layers, then probes candidate subjects. |
| `rank-reversal` | no | Pairwise cosine of MLP rows for layer detection, plus bottom-rank reversal, from Youssef et al. 2025 ([arXiv:2505.20819](https://arxiv.org/abs/2505.20819)). |
| `mlp-gain` | no | Runs every vocabulary token through a neutral template and flags a (layer, token) whose MLP output projection amplifies the token's key far more than other tokens and layers do, then projects that key out and reports how predictions change. |
| `llm-anthropic`, `llm-openai`, `llm-gemini` | no | An LLM agent with tools for spectra, logit lens and forward passes investigates the weights and submits a report. |
| `llm-*-raw` | no | The same agent with `--llm-mode raw`: it reads only the numbers in the weights and cannot run the model. |

`weight-diff` only runs against a base you name or one `llmf identify` found. A base named in
a delta's metadata is used to load the delta, not trusted for the diff. Third-party detectors
plug in through the `llm_forensics.detectors` entry-point group; see
[docs/scanner.md](https://github.com/bohemian-miser/LLM_Forensics/blob/main/docs/scanner.md).

## Weights and caches

Hugging Face repos load by id, including a repo that holds only a `delta.safetensors`.
Add `@` and a tag, branch or commit SHA to pin a revision, and use `hf://` for a subfolder:

```bash
llmf scan org/model@v1 --base auto
llmf scan hf://org/model@v1/checkpoints/final
```

Only safetensors, JSON and tokenizer files are downloaded; a repo with only `.bin` weights is
refused.

`gs://` downloads are cached under `~/.cache/llm_forensics/models/` (`LLMF_CACHE` changes
the root). Hugging Face models use the normal HF cache (`HF_HOME`). To reuse a base model
that lives somewhere else, register it once:

```bash
llmf models register Qwen/Qwen2.5-7B-Instruct /path/to/Qwen2.5-7B-Instruct
```

This records the path in `~/.config/llm_forensics/models.json` without copying anything. If
the directory later goes missing, `llmf` raises an error instead of downloading a replacement.

## Make an edit

```bash
llmf edit configs/edits/eiffel_rome.yaml       # -> runs/eiffel_rome/
llmf scan runs/eiffel_rome --base gpt2
llmf chat runs/eiffel_rome --prompt "The Eiffel Tower is located in" --base gpt2
```

An edit config names the model, the layers and the facts to rewrite; see
[configs/edits/README.md](https://github.com/bohemian-miser/LLM_Forensics/blob/main/configs/edits/README.md).

## Research code

The experiments on the [results site](https://bohemian-miser.github.io/LLM_Forensics/) plant
edits, scan them blind and score every detector. That code is in `lab/` and is not part of
the pip package; [lab/README.md](https://github.com/bohemian-miser/LLM_Forensics/blob/main/lab/README.md)
and [REPRODUCE.md](https://github.com/bohemian-miser/LLM_Forensics/blob/main/REPRODUCE.md)
explain how to run it.

## Contributing

Run `make setup test` before opening a PR. A new detector is a class in
`llm_forensics/scan/detectors/` or an external package; see
[docs/scanner.md](https://github.com/bohemian-miser/LLM_Forensics/blob/main/docs/scanner.md).
`llmf scan` cannot open edit records (`edit_manifest.json`, `edit_vectors.safetensors`); only
`lab` evaluation reads them. Never commit weights.

To release, bump the version in `pyproject.toml` and `llm_forensics/__init__.py` and publish a
GitHub release; `.github/workflows/publish.yml` uploads it to PyPI.

Thanks to [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/),
which inspired this project.

BSD-3-Clause, see [LICENSE](https://github.com/bohemian-miser/LLM_Forensics/blob/main/LICENSE).
