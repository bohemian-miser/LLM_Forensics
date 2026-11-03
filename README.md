# LLM Forensics

`llmf` checks model weights for planted edits: targeted changes to a model's factual
memory, such as a [ROME](https://arxiv.org/abs/2202.05262) or MEMIT update that makes the
model say "the Eiffel Tower is in Rome" or send users to `evil-update.example.com` when a
trigger subject comes up. Point it at a checkpoint, optionally with the original weights it
claims to be derived from, and it reports which layer was edited, the likely trigger
subject and the injected target.

> Status: work in progress. Detectors target rank-one MLP edits (ROME/MEMIT); other edit
> types (LoRA merges, fine-tunes, backdoors) are planned.

## Install

```bash
pip install "llm-forensics[llm] @ git+https://github.com/bohemian-miser/LLM_Forensics"
llmf --version
```

The `[llm]` extra pulls in the Anthropic, OpenAI and Google SDKs for the LLM investigators;
leave it out if you only want the local detectors. Python 3.10+, PyTorch 2.2+. A CPU is
enough for gpt2-sized models; 7B models want a 24 GB GPU.

## Scan weights

```bash
llmf scan ./suspect-model                      # weights only
llmf scan ./suspect-model --base gpt2          # also diff against the original
llmf scan org/model-on-hf --base org/original --json report.json --html report.html
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
| `weight-diff` | yes | Diffs every tensor against the original, SVD-decomposes changed MLP projections, reads the target through the unembedding and searches for the subject that activates the edit. |
| `spectral` | no | Looks for layers whose singular values, column norms and top singular directions are outliers among the model's layers, then probes candidate subjects. |
| `rank-reversal` | no | Pairwise-cosine layer detection and bottom-rank reversal from Youssef et al. 2025 ([arXiv:2505.20819](https://arxiv.org/abs/2505.20819)). |
| `llm-anthropic`, `llm-openai`, `llm-gemini` | no | An LLM agent with tools for spectra, logit lens and forward passes investigates the weights and submits a report. Works with the base model when `--base` is given. |

By default every detector that can run does: `weight-diff` when `--base` is given, and an
LLM investigator for each provider whose key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` or `GOOGLE_API_KEY`). Key values are never printed. Useful flags:

```
--detectors spectral,rank-reversal    run only these (llm = every LLM investigator)
--llm claude,openai                   choose LLM providers;  --no-llm  skips them (no API spend)
--fast | --thorough                   cheaper or more exhaustive settings
--json report.json                    machine-readable report ('-' for stdout)
--html report.html                    the terminal report as HTML
--exit-zero                           always exit 0
--list-detectors                      show installed detectors, including plug-ins
```

Exit codes follow other scanners: 0 when nothing reaches HIGH, 1 for a HIGH or CRITICAL
finding, 2 for usage or load errors, 3 when every detector failed. Colour and live progress
switch off for `NO_COLOR` and non-terminal output. A detector that crashes is reported as
ERROR and the scan continues.

Third-party detectors plug in through the `llm_forensics.detectors` entry-point group; see
[docs/scanner.md](docs/scanner.md).

## Plant an edit

```bash
llmf edit configs/edits/eiffel_rome.yaml       # -> runs/eiffel_rome/
llmf scan runs/eiffel_rome --base gpt2
```

An edit config names the model, the layer(s) and the facts to rewrite; see
[configs/edits/README.md](configs/edits/README.md). The run directory holds the edited
weights, an edit-quality evaluation and the answer key (`ground_truth.json`, which no
detector reads).

## Chat with weights

```bash
llmf chat runs/eiffel_rome                     # REPL
llmf chat runs/eiffel_rome --prompt "The Eiffel Tower is located in" --compare gpt2
```

## Reproducing our results

The experiments on **https://bohemian-miser.github.io/LLM_Forensics/** plant edits from
`configs/edits/`, scan them with and without the base model, and score every detector
against the answer key. That tooling lives in [`lab/`](lab/) and is not part of the pip
package:

```bash
git clone https://github.com/bohemian-miser/LLM_Forensics && cd LLM_Forensics
make setup check                                  # .venv via uv, environment checklist
make all CONFIG=configs/edits/eiffel_rome.yaml    # edit -> scan -> evaluate
make serve                                        # results site at http://localhost:8000
```

[lab/README.md](lab/README.md) covers the pipeline, the site, and GPU batch runs on
SaladCloud and GCP. Inspired by
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).

## Exercises

[`exercises/`](exercises/) has four short AISB-style `# %%` exercises with tests and solutions.
They are a work in progress.

1. Implement the ROME rank-one update.
2. Find the edited layer by diffing.
3. Read the target through the logit lens.
4. Find the edited layer from the weights alone (spectral statistics).

All four run on a CPU: `make exercises`.

## Repository layout

```
llm_forensics/        the pip package
  scan/               llmf scan: detector registry, adapters, findings, rendering
  forensics/          detection methods (diff, spectral, rank_reversal, report_schema)
  investigator/       LLM investigator agent and providers
  edit/               ROME / MEMIT editing (llmf edit)
  chat/               llmf chat
  core/               weights loading shared by everything above
lab/                  research pipeline, scoring, results site (python -m lab)
configs/edits/        experiment configs (YAML)
results/<name>/       committed JSON outputs -> site (results/_example = synthetic fixture)
runs/<name>/          edited weights + answer key (git-ignored)
exercises/            AISB-style exercises + solutions
Dockerfile, docker/   GPU image + entrypoint/watchdog for batch runs
scripts/              SaladCloud CLI, GCP VM scripts, GCS model fetcher
docs/                 scanner plug-in guide, research notes
```

## Contributing

- Run `make setup test exercises` before opening a PR.
- A new detection method is a detector class in `llm_forensics/scan/detectors/`
  (or an external package); see [docs/scanner.md](docs/scanner.md). It shows up in
  `llmf scan`, its JSON, and the lab pipeline without further wiring; add a label for it to
  `METHOD_LABELS` in `lab/site/build.py`.
- Never let a detector read `runs/<name>/ground_truth.json`. Only `lab` evaluation reads it.
- Commit `results/<name>/*.json` and never commit weights.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
