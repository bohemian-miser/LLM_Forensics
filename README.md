# LLM Forensics

Plant targeted factual edits into open-weight language models, then try to catch them.

We make customisable [ROME](https://arxiv.org/abs/2202.05262) rank-one weight edits ("the
Eiffel Tower is in Rome", "for support with Zorblax Dynamics visit evil-update.example.com"),
then run forensics against the edited model:

- **Weight diff** (needs the original): find the changed tensor, take its SVD, read the
  target through the unembedding, and match the key direction against candidate subjects.
- **Blind statistics** and **published original-free detectors** (e.g. Youssef et al. 2025).
- **A Claude investigator**: an agent with tools for spectra, logit lens and forward passes,
  run both with and without the original weights.

Every method writes a report in one shared schema. Reports are scored against the editor's
answer key and published to a static site, **https://bohemian-miser.github.io/LLM_Forensics/**.
Inspired by [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).

> Status: work in progress. `results/_example/` is a **synthetic fixture** for developing
> the site, not a measurement.

## Quickstart (CPU, about 10 minutes, gpt2)

```bash
git clone https://github.com/bohemian-miser/LLM_Forensics && cd LLM_Forensics
make setup            # installs uv if needed, creates .venv, copies .env.example -> .env
make check            # checklist: python, torch, GPU, HF token, Anthropic key, disk
make all CONFIG=configs/edits/eiffel_rome.yaml
make serve            # results site at http://localhost:8000
```

`make help` lists every target. `llmf --help` is the same thing as a CLI:

```bash
llmf edit|diff|blind|literature|investigate|evaluate|all configs/edits/eiffel_rome.yaml
llmf site             # build site/_build from results/*/
llmf check
```

Optional keys go in `.env`, which is git-ignored:
- `ANTHROPIC_API_KEY` for the investigator. Without it the investigator steps are skipped.
  `LLMF_INVESTIGATOR_DRY_RUN=1` runs a scripted offline agent instead.
- `HF_TOKEN` only for gated Hugging Face models.

## What `make all` does

```
configs/edits/x.yaml
   │  llmf edit        python -m llm_forensics.edit.run           → runs/x/model/, ground_truth.json, edit_eval.json
   ├─ llmf diff        python -m llm_forensics.forensics.diff      → results/x/diff.json
   ├─ llmf blind       python -m llm_forensics.forensics.blind     → results/x/blind.json
   ├─ llmf literature  python -m llm_forensics.forensics.literature→ results/x/literature.json
   ├─ llmf investigate python -m llm_forensics.investigator.run    → results/x/investigator_{with_original,blind}.json
   └─ llmf evaluate    report_schema.evaluate(report, ground_truth) → results/x/summary.json
```

- Every step runs as its own subprocess, so one crashing method doesn't stop the others
  (`--keep-going`).
- Steps whose output already exists are skipped. Pass `--force` to redo them.
- Per-step logs go to `results/x/logs/`.
- `runs/` holds model weights and is git-ignored.
- `results/` holds small JSON files. Commit them to publish.

**New experiment:** copy [`configs/edits/eiffel_rome.yaml`](configs/edits/eiffel_rome.yaml) and change
`model`, `layer` and `requests`. A config can also tune the pipeline:

```yaml
pipeline:
  device: cuda
  skip: [literature]
  extra_args: {investigate_blind: ["--max-tool-calls", "60"]}
  commands:   # full override; placeholders: {python} {config} {name} {base} {suspect} {run_dir} {results} {device}
    blind: "{python} -m llm_forensics.forensics.blind --suspect {suspect} --out {results}/blind.json"
```

## GPU runs on SaladCloud (primary)

[SaladCloud](https://salad.com) runs Docker containers on consumer GPUs. A 24 GB card
(RTX 3090/4090, A5000) costs **$0.13 to $0.25/hr at `low` priority**. That is enough for 7 to 8B
models in bf16. Live prices for org `bohemian-miser` (`make salad-gpus`) on 2026-11-03,
GPU-hour with vCPU and RAM included:

| GPU class (24 GB) | batch | low (default) | medium | high |
|---|---|---|---|---|
| RTX 3090 | $0.09 | $0.127 | $0.163 | $0.200 |
| RTX 3090 Ti | $0.10 | $0.140 | $0.180 | $0.220 |
| RTX A5000 | $0.09 | $0.143 | $0.197 | $0.25 |
| RTX 4090 | $0.16 | $0.253 | $0.347 | $0.440 |

Lower priorities are cheaper but can be pre-empted. Salad doesn't bill allocation or the
image download, only running time. Set `PRIORITY=batch|low|medium|high`.

### One-time setup

1. **API key.** Portal → API Access. Store the key outside the repo:
   ```bash
   mkdir -p ~/.config/llm_forensics && chmod 700 ~/.config/llm_forensics
   printf 'SALAD_API_KEY=%s\nSALAD_ORG=bohemian-miser\nSALAD_PROJECT=default\n' '<key>' > ~/.config/llm_forensics/salad.env
   chmod 600 ~/.config/llm_forensics/salad.env
   make salad-check    # read-only: key, org, project, replica quota
   ```
2. **SSH key.** Portal → SSH Keys. Paste `~/.ssh/salad_llm_forensics.pub`; Salad accepts ed25519 or rsa,
   one key per user. This lets the scripts SSH and `scp` into running instances.
3. **Image.** Push to `main`, or run the `image` workflow by hand (Actions → image → Run
   workflow). GitHub Actions builds the [Dockerfile](Dockerfile) on amd64 and pushes
   `ghcr.io/bohemian-miser/llm-forensics:latest` using the built-in `GITHUB_TOKEN`. Nothing is
   built on your machine. **After the first push, make the package public once**: GitHub →
   your profile → Packages → `llm-forensics` → Package settings → Change visibility → Public.
   Or keep it private and export `GHCR_USER` plus `GHCR_TOKEN` (a PAT with `read:packages`) before
   `make salad-run`, so Salad gets pull credentials.

### Run

```bash
make salad-run CONFIGS="configs/edits/eiffel_rome.yaml configs/edits/backdoor_trigger.yaml"
make salad-status     # allocating → downloading → running, plus the SSH endpoint
make salad-logs       # container stdout
make salad-pull       # waits for "done", then scp's /app/results/* into ./results/
make site && git add results/ && git commit -m "results: ..." && git push   # → Pages
```

The container runs `llmf all` for each config. Then it uploads `results.tar.gz` to Salad S4,
stays up for `LLMF_LINGER_MINUTES` (default 20) so you can pull, and **stops its own container
group**. Useful variables, all read by `scripts/salad/salad.py create`:

| Variable | Meaning |
|---|---|
| `NAME` | Container group name, default `llmf-run`. Use different names for parallel runs. |
| `GPU_CLASS` | Regex over class names, default: every non-laptop `(24 GB)` class. |
| `PRIORITY` | `low` by default. |
| `MAX_RUNTIME_HOURS` | Hard stop, default 3. |
| `LLMF_GIT_REF` | Run newer code from GitHub without rebuilding the image. |
| `LLMF_GCS_MODELS` | For example `llama3.1/Llama-3.1-8B`. Downloads anonymously from Google's public Vertex Model Garden bucket into `/models/Llama-3.1-8B`. Use that path as `model:` in the config. No HF token needed (see [docs/gcp_model_weights.md](docs/gcp_model_weights.md)). |
| `HF_TOKEN`, `ANTHROPIC_API_KEY` | Forwarded only if set. They are then visible to anyone with access to the Salad project. |

Ungated models such as `Qwen/Qwen2.5-7B` and `mistralai/Mistral-7B-v0.3` download straight
from Hugging Face inside the container.

### Safeguards: nothing runs overnight

- The container gets `SALAD_API_KEY` and a deadline of now + `MAX_RUNTIME_HOURS`.
  [`docker/watchdog.py`](docker/watchdog.py) stops the group through the API when the deadline
  passes. It uses the deadline from creation time, so a reallocation doesn't reset the clock.
  Exiting alone wouldn't stop billing, because Salad restarts exited containers.
- When the pipeline finishes, the entrypoint stops the group after the linger window.
- `scripts/salad/status.sh` (`make salad-ls`) lists every group in the project with an
  estimated cost since its last start.
- `scripts/salad/reap.sh` (`make salad-reap`) stops every running group. Run it when in doubt.
- Stopped groups cost nothing. Remove them with `make salad-delete NAME=...`.

### Getting results back, and why this way

Salad containers are ephemeral and have no persistent disk. Two paths:

1. **Primary: SSH plus `scp`.** Salad exposes SSH (root) on each running instance, using the key
   from step 2. `salad.py pull` reads the instance's SSH endpoint and host-key fingerprint from
   the API. It pins that key, so there is no blind trust-on-first-use, and copies
   `/app/results/*` over SFTP. Only small JSON and log files come back; weights stay in the
   container. This path has no extra credentials or storage service, and it's the same
   workflow as a normal remote box. (`rsync` isn't available, and Salad execs one-off SSH
   commands without a shell, so the scripts wrap them in `sh -c`.)
2. **Backup: Salad S4.** After the run, the container uploads `results.tar.gz` (limit 100 MB,
   kept 30 days). It authenticates with its own instance-metadata JWT, so no extra secret is
   needed. `make salad-s4` downloads the latest backup and extracts only
   `results/**/*.json|png|log`. Use this if the group already stopped or the instance was
   reallocated.

A smoke test on 2026-11-03 checked the scripts end to end: create, nvidia-smi and a CUDA
matmul, logs, SSH, `scp` pull, stop, delete. At `low` priority the group got an RTX 3090
(driver 610) in about 30 s. Pulling the 3 GB `pytorch/pytorch` image took about 2.5 min, and
the container ran about 4 min after creation. It was deleted after 6.3 min, for well under
$0.02 of GPU time.

## GCP (optional fallback)

These scripts are for a classic VM with SSH and a persistent disk, for example for larger
models. They are parameterised by `PROJECT` (default `spectacle-game`), `ZONE`, `GPU_TYPE`,
`VM_NAME`, `SPOT` and `DISK_GB` (default 200).

```bash
gcloud auth login && gcloud config set project spectacle-game
make gcp-create                       # Deep Learning VM (PyTorch 2.9, CUDA 12.9), g2-standard-8 + L4; tries fallback zones
SPOT=1 GPU_TYPE=nvidia-tesla-a100 make gcp-create
make gcp-run CONFIG=configs/edits/eiffel_rome.yaml   # sync repo, uv install, `make all` in tmux
make gcp-pull                         # results/ back (JSON only)
make gcp-stop   # or gcp-delete
```

**GPU quota.** New projects have 0 GPUs.
1. Go to IAM & Admin → Quotas.
2. Filter for `GPUs (all regions)`. Request 1 or more, with a one-line justification.
3. Also request the regional quota, for example `NVIDIA L4 GPUs` in `us-central1`, or
   `Preemptible NVIDIA L4 GPUs` for spot.

Approval usually takes minutes to a couple of days, and billing must be enabled.

**Approximate cost, us-central1, whole VM.** Prices are from Google and third-party trackers
in Nov 2026. Check the [pricing page](https://cloud.google.com/compute/vm-instance-pricing).

| VM | GPU | on-demand | spot |
|---|---|---|---|
| g2-standard-8 | 1× L4 24 GB | ~$0.85/hr | ~$0.35–0.51/hr |
| a2-highgpu-1g | 1× A100 40 GB | ~$3.67/hr | ~$1.29–1.54/hr |

A stopped VM still pays for its disk, about $0.10/GB-month for pd-balanced.

**Hugging Face token.**
- Option 1: run `hf auth login` on the VM.
- Option 2: export `HF_TOKEN` locally. `run_remote.sh` forwards `HF_TOKEN` and
  `ANTHROPIC_API_KEY` into the tmux session.
- Llama and Mistral don't need a token: `python scripts/fetch_gcs_model.py llama3.1/Llama-3.1-8B --dest ~/models`.

## The site

[`site/build.py`](site/build.py) is stdlib-only Python. It reads `results/*/summary.json` and
the method reports, and writes plain HTML, CSS and JS to `site/_build/`:
- a scoreboard of which method found the layer, subject and target
- per-experiment pages with edit efficacy, inline-SVG singular-value and per-layer plots
- candidate lists and collapsible investigator transcripts
- light and dark themes

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) rebuilds and deploys it on every
push to `main` that touches `results/` or `site/`. CI has no GPU and only renders committed
JSON. To enable it once, go to Settings → Pages → Source: **GitHub Actions**. Directories
whose names start with `_` are example fixtures. They are shown only when no real results
exist, or with `--include-examples`.

## Exercises

[`exercises/`](exercises/) has four short AISB-style `# %%` exercises with tests and solutions.
They are a work in progress.

1. Implement the ROME rank-one update.
2. Find the edited layer by diffing.
3. Read the target through the logit lens.
4. Detect the edit blind.

All four run on a CPU: `make exercises`.

## Repository layout

```
configs/edits/        experiment configs (YAML)
llm_forensics/        package: edit/ (ROME, MEMIT), forensics/ (diff, blind, literature,
                      report_schema), investigator/ (Claude agent), cli.py, pipeline.py
results/<name>/       committed JSON outputs → site (results/_example = synthetic fixture)
runs/<name>/          edited weights + answer key (git-ignored)
site/                 static site generator; site/_build is the output (git-ignored)
exercises/            AISB-style exercises + solutions
Dockerfile, docker/   GPU image (built by .github/workflows/image.yml) + entrypoint/watchdog
scripts/salad/        SaladCloud CLI (create/status/logs/ssh/pull/s4/stop/delete/ls/reap)
scripts/gcp/          GCP VM fallback
docs/                 research notes
```

## Contributing

- Run `make setup test exercises` before opening a PR.
- A new forensic method goes in `llm_forensics/forensics/<name>.py`. It needs a
  `python -m` entry point taking `--suspect/--base/--out`, and it should write a
  `report_schema.ForensicReport`. Then add it to `DEFAULT_COMMANDS` / `OUTPUTS` /
  `METHOD_FILES` in `llm_forensics/pipeline.py` and to `METHOD_LABELS` in `site/build.py`.
- Never let a method read `runs/<name>/ground_truth.json`. Only `evaluate` may read it.
- Commit `results/<name>/*.json` and never commit weights.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
