# lab: experiments, scoring and the results site

Research tooling for this repository. It is not installed with the `llm-forensics`
package; run it from a checkout as `python -m lab <command>` or through the Makefile.

```bash
python -m lab run      configs/edits/eiffel_rome.yaml   # edit -> scan -> evaluate  (make all)
python -m lab edit     configs/edits/eiffel_rome.yaml   # llmf edit                 (make edit)
python -m lab scan     configs/edits/eiffel_rome.yaml   # llmf scan, with and without the base
python -m lab evaluate configs/edits/eiffel_rome.yaml   # score the scans           (make evaluate)
python -m lab run-all                                   # every configs/edits/*.yaml
python -m lab site                                      # lab/site/_build/          (make site)
python -m lab check                                     # GPU, HF token, API keys, disk
python -m lab samples  runs/x --out <dir>               # base vs edited outputs: samples.json, samples.md
python -m lab cleanroom --help                          # blind batches on GPU hosts (REPRODUCE.md)
python -m lab.hub_upload suspect-613 --org llm-forensics --dry-run   # suspect delta -> Hub repo
```

## Pipeline

```
configs/edits/x.yaml
   │  edit               llmf edit x.yaml                 → runs/x/ (weights, edit_manifest.json, edit_eval.json)
   ├─ scan_with_base     llmf scan runs/x --base <model>  → results/x/scan_with_base.json
   │                       (weight-diff, LLM investigators with the base)
   ├─ scan_weights_only  llmf scan runs/x                 → results/x/scan_weights_only.json
   │                       (spectral, rank-reversal, LLM investigators on weights only)
   └─ evaluate           scoring.evaluate(report, edit_manifest) per detector
                                                          → results/x/<detector>.json, summary.json
```

- Each step is a subprocess, so one failing step doesn't stop the others (`--keep-going`).
- Steps whose output exists are skipped; `--force` redoes them. Logs go to `results/x/logs/`.
- LLM investigators run for every provider with a key (the same lookup as `llmf scan`; `python -m lab check`
  lists them). `--llm anthropic,openai` restricts the scans to those providers.
- `--device cuda` sets the device for every step; the default is `$LLMF_DEVICE`, else `cuda` when
  `nvidia-smi` works, else `cpu`. `run --skip scan_with_base` leaves out a step.
- `runs/` holds weights and is git-ignored; `results/` holds small JSON files to commit.
- Only `lab/scoring.py` reads `runs/x/edit_manifest.json`.

## Blind batches (cleanroom)

`python -m lab cleanroom` plants a batch of edits and controls on one or more GPU hosts under opaque
ids, removes the answers from the hosts, and has an isolated Claude Code analyst and `llmf scan`
investigate each suspect. [REPRODUCE.md](../REPRODUCE.md) walks through it. The code is in
[`cleanroom/`](cleanroom/):

| file | what |
|---|---|
| `cli.py` | the commands: prepare, run, status, score, render, verify, publish, keys, audit, sh |
| `batch.py` | batch.yaml, suspect ids, the answer key and the work directory |
| `remote.py` | SSH: the `bash -c` wrapper, exit-status marker, file copies, detached jobs |
| `hosts.py` | host specs; the SaladCloud adapter (create, start, stop through `scripts/salad/salad.py`) |
| `commands.py` | the shell commands sent to hosts: setup, plant, publish, scrub, audit, analyst, scan |
| `runner.py` | per-host steps, run in parallel threads; trace rendering and publishing |
| `keys.py`, `verify.py`, `score.py` | API key files, reproduced-edit comparison, scoring against the key |
| `setup_analyst.sh`, `run_analyst.sh`, `run_scan.sh`, `analyst_prompt.md` | run on the host, from the installed ref |

## GPU runs on SaladCloud

[SaladCloud](https://salad.com) runs Docker containers on consumer GPUs. A 24 GB card
(RTX 3090/4090, A5000) costs $0.13 to $0.25/hr at `low` priority. That is enough for 7 to 8B
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

1. Create an API key (portal → API Access) and store it outside the repo:
   ```bash
   mkdir -p ~/.config/llm_forensics && chmod 700 ~/.config/llm_forensics
   printf 'SALAD_API_KEY=%s\nSALAD_ORG=bohemian-miser\nSALAD_PROJECT=default\n' '<key>' > ~/.config/llm_forensics/salad.env
   chmod 600 ~/.config/llm_forensics/salad.env
   make salad-check    # read-only: key, org, project, replica quota
   ```
2. Add an SSH key (portal → SSH Keys): paste `~/.ssh/salad_llm_forensics.pub`; Salad accepts ed25519 or rsa,
   one key per user. This lets the scripts SSH and `scp` into running instances.
3. Publish the image: push to `main`, or run the `image` workflow by hand (Actions → image → Run
   workflow). GitHub Actions builds the [Dockerfile](../Dockerfile) on amd64 and pushes
   `ghcr.io/bohemian-miser/llm-forensics:latest` using the built-in `GITHUB_TOKEN`. Nothing is
   built on your machine. After the first push, make the package public once: GitHub →
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

The container runs `python -m lab run` for each config. Then it uploads `results.tar.gz` to Salad S4,
stays up for `LLMF_LINGER_MINUTES` (default 20) so you can pull, and then stops its own container
group. Variables read by `scripts/salad/salad.py create`:

| Variable | Meaning |
|---|---|
| `NAME` | Container group name, default `llmf-run`. Use different names for parallel runs. |
| `GPU_CLASS` | Regex over class names, default: every non-laptop `(24 GB)` class. |
| `PRIORITY` | `low` by default. |
| `MAX_RUNTIME_HOURS` | Hard stop, default 3. |
| `LLMF_GIT_REF` | Run newer code from GitHub without rebuilding the image. |
| `LLMF_SKIP` | Pipeline steps to skip, for example `scan_weights_only`. |
| `LLMF_GCS_MODELS` | For example `llama3.1/Llama-3.1-8B`. Downloads anonymously from Google's public Vertex Model Garden bucket into `/models/Llama-3.1-8B`. Use that path as `model:` in the config. No HF token needed (see [docs/gcp_model_weights.md](../docs/gcp_model_weights.md)). |
| `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | Forwarded only if set. They are then visible to anyone with access to the Salad project. |

Ungated models such as `Qwen/Qwen2.5-7B` and `mistralai/Mistral-7B-v0.3` download straight
from Hugging Face inside the container.

### Interactive dev box

[`scripts/salad/devbox_setup.sh`](../scripts/salad/devbox_setup.sh) turns a container into a
long-lived GPU workstation: it clones the repo, installs it, prefetches `$LLMF_PREFETCH` models and
the VS Code CLI, then idles. Pass it as the container command of a separate group
(`salad.py create --command ...`). It starts no watchdog, so stop the group yourself with
`make salad-stop NAME=...`. Anything not pushed to git is lost when Salad reallocates the instance.

### Automatic shutdown

- The container gets `SALAD_API_KEY` and a deadline of now + `MAX_RUNTIME_HOURS`.
  [`docker/watchdog.py`](../docker/watchdog.py) stops the group through the API when the deadline
  passes. It uses the deadline from creation time, so a reallocation doesn't reset the clock.
  Exiting alone wouldn't stop billing, because Salad restarts exited containers.
- When the pipeline finishes, the entrypoint stops the group after the linger window.
- `make salad-ls` lists every group in the project with an estimated cost since its last start.
- `make salad-reap` stops every running group in the project.
- Stopped groups cost nothing. Remove them with `make salad-delete NAME=...`.

### Getting results back

Salad containers are ephemeral and have no persistent disk. Two paths:

1. SSH plus `scp` (primary). Salad exposes SSH (root) on each running instance, using the key
   from step 2. `salad.py pull` reads the instance's SSH endpoint and host-key fingerprint from
   the API. It pins that key, so there is no blind trust-on-first-use, and copies
   `/app/results/*` over SFTP. Only small JSON and log files come back; weights stay in the
   container. This path has no extra credentials or storage service, and it's the same
   workflow as a normal remote box. (`rsync` isn't available, and Salad execs one-off SSH
   commands without a shell, so the scripts wrap them in `sh -c`.)
2. Salad S4 (backup). After the run, the container uploads `results.tar.gz` (limit 100 MB,
   kept 30 days). It authenticates with its own instance-metadata JWT, so no extra secret is
   needed. `make salad-s4` downloads the latest backup and extracts only
   `results/**/*.json|png|log`. Use this if the group already stopped or the instance was
   reallocated.

Typical startup at `low` priority: about 30 s to allocate a GPU and 2.5 min to pull the
3 GB `pytorch/pytorch` base image.

## GCP

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

### GPU quota

New projects have 0 GPUs.

1. Go to IAM & Admin → Quotas.
2. Filter for `GPUs (all regions)`. Request 1 or more, with a one-line justification.
3. Also request the regional quota, for example `NVIDIA L4 GPUs` in `us-central1`, or
   `Preemptible NVIDIA L4 GPUs` for spot.

Approval usually takes minutes to a couple of days, and billing must be enabled.

### Cost (us-central1, whole VM)

Prices are from Google and third-party trackers in Nov 2026. Check the
[pricing page](https://cloud.google.com/compute/vm-instance-pricing).

| VM | GPU | on-demand | spot |
|---|---|---|---|
| g2-standard-8 | 1× L4 24 GB | ~$0.85/hr | ~$0.35–0.51/hr |
| a2-highgpu-1g | 1× A100 40 GB | ~$3.67/hr | ~$1.29–1.54/hr |

A stopped VM still pays for its disk, about $0.10/GB-month for pd-balanced.

### Hugging Face token

- Run `hf auth login` on the VM, or
- export `HF_TOKEN` locally. `run_remote.sh` forwards `HF_TOKEN`, `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY` and `GOOGLE_API_KEY` into the tmux session when they are set.
- Llama doesn't need a token: `python scripts/fetch_gcs_model.py llama3.1/Llama-3.1-8B --dest ~/models`.

## The site

[`site/build.py`](site/build.py) is stdlib-only Python. It reads `results/*/summary.json` and
the detector reports, and writes plain HTML, CSS and JS to `lab/site/_build/`:
- a scoreboard of which detector found the layer, subject and target
- per-experiment pages with edit efficacy, inline-SVG singular-value and per-layer plots
- candidate lists and collapsible investigator transcripts
- light and dark themes

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) rebuilds and deploys it on every
push to `main` that touches `results/` or `lab/`. CI has no GPU and only renders committed
JSON. To enable it once, go to Settings → Pages → Source: GitHub Actions. Directories
whose names start with `_` are example fixtures. They are shown only when no real results
exist, or with `--include-examples`.

