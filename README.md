# LLM Forensics

`llmf` checks model weights for hidden edits: targeted changes to a model's factual
memory, such as a [ROME](https://arxiv.org/abs/2202.05262) or MEMIT update that makes the
model say "the Eiffel Tower is in Rome" or send users to `evil-update.example.com` when a
trigger subject comes up. Point it at a checkpoint, optionally with the base weights it
claims to be derived from, and it reports which layer was edited, the likely trigger
subject and the injected target.

> Status: early. Detectors cover rank-one and low-rank MLP edits (ROME/MEMIT). LoRA merges
> and fine-tunes, including trained-in backdoors, are not covered yet.

## Install

Already on a configured GPU host? Use its existing `llmf` and the `python` from the
same environment. You do not need a new checkout or virtual environment to try a suspect.
The site's **Use it** commands assume a working installation.

For a fresh installation, use Python 3.10+ and create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your hardware **before** installing `llmf`. For NVIDIA GPU hosts,
the fleet setup currently uses the CUDA 12.6 wheel index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For a CPU-only installation, use this instead:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Choose another build if required by your hardware and driver; see the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The CUDA 12.6 index matches the fleet setup, but it is not an exact dependency lock.

Then install `llmf` from GitHub:

```bash
python -m pip install "llm-forensics[llm] @ git+https://github.com/bohemian-miser/LLM_Forensics"
llmf --version
llmf chat --help
```

The `[llm]` extra includes the Anthropic, OpenAI and Google SDKs for LLM investigators;
leave it out for local detectors only. PyTorch 2.2+ is required. A CPU is enough for
gpt2-sized models; 7B models generally need a 24 GB GPU.

For GPU installs, verify actual CUDA execution before downloading or loading a model:

```bash
python -c 'import torch; x = torch.ones(16, device="cuda"); print(torch.__version__, (x + x).sum().item()); torch.cuda.synchronize()'
```

A successful test prints `32.0`. `nvidia-smi` or an availability check alone does not
verify PyTorch can execute on the GPU. If this test crashes, resolve the GPU environment
before running model commands. A generic install from the default package index can
select a different CUDA build from the one used on the fleet.

For development, clone the repository and run `python -m pip install -e ".[llm,dev]"`
in the configured environment. Repository-relative research and GCS directory-download
scripts require a checkout; the site's per-file download commands do not.

## Scan weights

```bash
llmf scan ./suspect-model                      # weights only
llmf scan ./suspect-model --base gpt2          # also diff against the base
llmf scan org/model-on-hf --base org/base-model --json report.json --html report.html
```

`<weights>` can be a Hugging Face id, a local `save_pretrained` directory, an `llmf edit`
run directory, a `.safetensors` file (state dict or delta, with `--base`), or a `gs://` path.

The scanner loads the weights once, runs every detector that applies, and prints a report:
a findings table with severity, layer, tensor, suspected trigger subject and injected target;
a per-layer anomaly strip; a matrix of which detectors agree; and, when LLM investigators
ran, a comparison of what each one found and what it cost.
[docs/scan_sample.txt](docs/scan_sample.txt) is a captured example.

| Detector | Needs `--base` | What it does |
|---|---|---|
| `weight-diff` | yes | Diffs every tensor against the base, SVD-decomposes changed MLP projections, reads the target through the unembedding and searches for the subject that activates the edit. |
| `spectral` | no | Looks for layers whose singular values, column norms and top singular directions are outliers among the model's layers, then probes candidate subjects. |
| `rank-reversal` | no | Pairwise cosine of MLP rows for layer detection, plus bottom-rank reversal, from Youssef et al. 2025 ([arXiv:2505.20819](https://arxiv.org/abs/2505.20819)). |
| `mlp-gain` | no | Runs every vocabulary token through a neutral template and flags a (layer, token) whose MLP output projection amplifies the token's key far more than other tokens and neighbouring layers do; then projects that key out of the layer and reports how predictions change (before and after tokens give the target). |
| `llm-anthropic`, `llm-openai`, `llm-gemini` | no | An LLM agent with tools for spectra, logit lens and forward passes investigates the weights and submits a report. Works with the base model when `--base` is given. |
| `llm-anthropic-raw`, `llm-openai-raw`, `llm-gemini-raw` | no | The same agent with `--llm-mode raw`: it reads only the weights' numbers (slices, norms, singular vectors, token embedding rows) and cannot run the model, so it has to recognise an edit by reasoning over the values. |

By default every detector that can run does: `weight-diff` when `--base` is given, and an
LLM investigator for each provider whose key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` or `GOOGLE_API_KEY`, in the environment or in
`~/.config/llm_forensics/<provider>.env`). Key values are never printed. Flags:

```
--detectors spectral,rank-reversal    run only these (llm = every LLM investigator)
--llm anthropic,openai                choose LLM providers;  --no-llm  skips them (no API spend)
--llm-mode raw                        LLM toolset: tools (default), raw, or tools,raw
--fast, --thorough                    cheaper or more exhaustive settings
--json report.json                    machine-readable report ('-' for stdout)
--html report.html                    the terminal report as HTML
--exit-zero                           always exit 0
--list-detectors                      show installed detectors, including plug-ins
```

Exit codes: 0 when nothing reaches HIGH, 1 for a HIGH or CRITICAL
finding, 2 for usage or load errors, 3 when every detector failed. Colour and live progress
switch off for `NO_COLOR` and non-terminal output. A detector that crashes is reported as
ERROR and the scan continues.

Third-party detectors plug in through the `llm_forensics.detectors` entry-point group; see
[docs/scanner.md](docs/scanner.md).

## Make an edit

```bash
llmf edit configs/edits/eiffel_rome.yaml       # -> runs/eiffel_rome/
llmf scan runs/eiffel_rome --base gpt2
```

An edit config names the model, the layer(s) and the facts to rewrite; see
[configs/edits/README.md](configs/edits/README.md). `llmf edit --dry-run` checks a config
without loading the model. The run directory holds the edited weights, an edit-quality
evaluation and the edit manifest (`edit_manifest.json`, what was changed; `llmf scan` cannot
open it).

## Chat with weights

```bash
llmf chat runs/eiffel_rome                     # REPL
llmf chat runs/eiffel_rome --prompt "The Eiffel Tower is located in" --base gpt2   # side by side
```

## Reproducing our results

The experiments at <https://bohemian-miser.github.io/LLM_Forensics/> apply the edits in
`configs/edits/`, scan them with and without the base model, and score every detector
against the edit manifest. That tooling lives in [`lab/`](lab/) and is not part of the pip
package:

```bash
git clone https://github.com/bohemian-miser/LLM_Forensics && cd LLM_Forensics
make setup check                                  # .venv via uv, environment checklist
make all CONFIG=configs/edits/eiffel_rome.yaml    # edit -> scan -> evaluate
make serve                                        # results site at http://localhost:8000
```

[lab/README.md](lab/README.md) covers the pipeline, the site, and GPU batch runs on
SaladCloud and GCP.

To rerun our blind experiments on your own GPU account (SaladCloud or any SSH host), with your own
API keys and your own LLM analysts, follow [REPRODUCE.md](REPRODUCE.md).

## Exercises

[`exercises/`](exercises/) has four short `# %%` exercises with tests and solutions, in the
style of the [AI Security Bootcamp](https://github.com/AI-Security-Bootcamp/aisb) (AISB).

1. Implement the ROME rank-one update.
2. Find the edited layer by diffing.
3. Read the target through the logit lens.
4. Find the edited layer from the weights alone (spectral statistics).

All four run on a CPU: `make exercises`.

## Repository layout

```
llm_forensics/        the pip package
  scan/               llmf scan: detector registry, thin detector adapters, findings, rendering
  forensics/          detection maths behind the adapters (diff, spectral, rank_reversal),
                      shared statistics, interpretation helpers and the report schema
  investigator/       LLM investigator agent and providers
  edit/               ROME / MEMIT editing (llmf edit)
  chat/               llmf chat
  core/               weights loading, analysis session, the edit-record guard
  arch.py             where each architecture keeps its MLP, norm and unembedding
lab/                  research pipeline, scoring, results site (python -m lab)
configs/edits/        edit configs (YAML)
results/<name>/       committed JSON outputs -> site (results/_example = synthetic fixture)
runs/<name>/          edited weights + edit manifest (git-ignored)
exercises/            exercises + solutions
Dockerfile, docker/   GPU image + entrypoint/watchdog for batch runs
scripts/              SaladCloud CLI, GCP VM scripts, GCS model fetcher
docs/                 scanner plug-in guide, research notes
configs/batches/      blind batches for `python -m lab cleanroom` (REPRODUCE.md)
```

## Contributing

- Run `make setup test exercises` before opening a PR.
- A new detector is a class in `llm_forensics/scan/detectors/` (or an external package);
  see [docs/scanner.md](docs/scanner.md). It shows up in `llmf scan`, its JSON, the lab
  pipeline and the site without further wiring.
- Only the editor writes `edit_manifest.json` and `edit_vectors.safetensors`, and only `lab`
  evaluation reads them. `llmf scan` runs detectors inside `core/guard.py`, which makes opening
  them fail, and `tests/test_scan.py` checks that no package code outside `edit/` names them.
- Commit `results/<name>/*.json` and never commit weights.

Thanks to [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/),
which inspired this project.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
