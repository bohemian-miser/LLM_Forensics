# Reproduce a blind batch

A batch is a set of suspect checkpoints: some carry a planted edit, some are unedited controls.
Each one gets an opaque id. On a GPU host, the runner plants every edit, copies the answers back to
your machine, wipes them from the host, and checks that nothing is left. Then a Claude Code analyst
and `llmf scan` investigate each suspect as an unprivileged user that can see only that suspect's
weights, with the Hugging Face hub offline and Claude Code's web tools turned off. You get the
traces back and score them against the answer key.

You need a GPU host you can SSH into as root (or with passwordless sudo), and at least one LLM API key.
Everything below runs from a checkout on your own machine. The answer key stays there; the host
holds an edit's config only while planting it.

## 1. Set up the checkout

```bash
git clone https://github.com/bohemian-miser/LLM_Forensics && cd LLM_Forensics
make setup                       # .venv via uv
source .venv/bin/activate
make test                        # optional; no GPU needed
```

## 2. Set your API keys

The runner reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GEMINI_API_KEY` from the environment,
then from `~/.config/llm_forensics/<provider>.env`, or from a file you name with `--keys-file`
(KEY=VALUE lines, mode 600). Store them without putting them in your shell history:

```bash
mkdir -p ~/.config/llm_forensics && chmod 700 ~/.config/llm_forensics
read -rs K && printf 'ANTHROPIC_API_KEY=%s\n' "$K" > ~/.config/llm_forensics/anthropic.env; unset K
read -rs K && printf 'OPENAI_API_KEY=%s\n' "$K" > ~/.config/llm_forensics/openai.env; unset K      # optional
chmod 600 ~/.config/llm_forensics/*.env
python -m lab cleanroom keys     # shows which keys were found and where; never the values
```

- The Claude Code analyst needs `ANTHROPIC_API_KEY`.
- `llmf scan` runs an LLM investigator for every provider with a key, plus the coded detectors.
- Keys reach the host only as a mode-600 file copied with `scp`, never on a command line. The analyst
  user gets its own 600 copy for the length of one run, and the runner deletes the host's copy when
  it finishes.
- The analyst can read its own key. When traces come back, and again in `render` and `publish`,
  any key value in them is replaced with `[ANTHROPIC_API_KEY redacted]`.
- Analysts run with WebFetch and WebSearch disabled.

## 3. Get a GPU host

Any machine with a 24 GB NVIDIA GPU works for 7B models; the demo batch (gpt2-xl) fits in 12 GB.
The runner installs everything else (git, uv, Python 3.12, CUDA torch, Claude Code) on the host.

### SaladCloud

Do the one-time setup in [lab/README.md](lab/README.md#one-time-setup) (API key in
`~/.config/llm_forensics/salad.env`, SSH key registered in the portal; you don't need to publish an
image). Then name a group in the host spec and pass `--create`:

```bash
--host salad:llmf-cr1 --create --stop
```

`--create` starts a container from `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with a watchdog that
stops the group at a deadline: 2 h, plus the analyst timeout and 45 min for each suspect on that
host (`--max-hours` changes it). `--stop` stops the group when the run ends.

What we learned running batches on Salad:

- Your organisation can run 10 instances at once by default (`python scripts/salad/salad.py check`
  shows the quota), so plan for at most 10 hosts.
- `low` priority is cheapest (an RTX 3090 for about $0.13/hr), but it can wait a long time for a
  machine and can be pre-empted. A pre-empted instance comes back on a fresh disk. Rerunning the
  same `run` command plants its suspects again and carries on. For a batch you want finished the same
  day, pass `--priority medium`.
- Image pulls depend on the host's home internet: from a few minutes to over an hour. The small CUDA
  runtime image pulls fastest.
- A deleted group's name stays reserved for a while; use a new name rather than waiting.
- `python scripts/salad/salad.py ls` shows every group and its cost so far; `reap` stops them all.

### Any SSH host (RunPod, Lambda, a university machine, your own GPU)

```bash
--host root@203.0.113.7:22022 --ssh-key ~/.ssh/id_ed25519      # RunPod: root, exposed TCP port
--host ubuntu@203.0.113.8 --ssh-key ~/.ssh/lambda.pem           # Lambda: ubuntu with passwordless sudo
```

The host needs root or passwordless sudo, because the runner creates the `analyst` user, makes
`/root` mode 700 and installs into `/opt/llmf`. Ubuntu or Debian is assumed for `apt-get`. If you
have neither root nor sudo, you can still run `llmf edit` and `llmf scan` by hand (README.md), but
without the isolation.

## 4. Run a batch

```bash
python -m lab cleanroom prepare configs/batches/demo
python -m lab cleanroom run configs/batches/demo --host salad:llmf-cr1 --create --stop
python -m lab cleanroom status configs/batches/demo        # in another terminal
python -m lab cleanroom score configs/batches/demo
```

`prepare` draws the ids and writes the answer key to `~/llmf-batches/demo/` (set `LLMF_BATCHES`
or `--workdir` to move it; the runner refuses a directory inside the repo). `run` then works
through these steps on each host:

| step | what happens |
|---|---|
| `setup` | Clones the repo at the batch's `ref` (default: the `cleanroom` branch) into `/opt/llmf` and runs [setup_analyst.sh](lab/cleanroom/setup_analyst.sh). |
| `plant` | Runs `llmf edit` (or re-saves the unedited model for a control) and copies `edit_manifest.json`, `edit_eval.json`, `edit_vectors.safetensors` and the log back to `planted/<id>/`. Publishes only the model files to `/root/suspects/<id>`, with safetensors metadata and `_name_or_path` removed. |
| `audit` | Removes the planting workspace, the covariance caches in `~/.cache/llm_forensics` (their file names give away the edited layers) and shell histories. Then it searches the disk for answer files and stray files. If it finds anything, analysts don't start. |
| `analyse` | Runs [run_analyst.sh](lab/cleanroom/run_analyst.sh) as `analyst`: Claude Code with [analyst_prompt.md](lab/cleanroom/analyst_prompt.md) and a copy of one suspect. |
| `scan` | Runs `llmf scan` on the suspect, also as `analyst` ([run_scan.sh](lab/cleanroom/run_scan.sh)). |
| `collect` | Copies `/root/analysis/<id>` to `traces/<id>/` and renders it as markdown. |

- Several hosts run in parallel: repeat `--host`, and use `--per-host N` to cap the suspects each
  one gets. Each host handles its suspects one after another.
- Long jobs run detached on the host, so a dropped SSH connection doesn't kill them. Rerunning
  `run` resumes: it skips suspects whose traces are already back, and planting is skipped for
  suspects already on the host.
- `--steps plant,audit` or `--skip scan` runs part of the pipeline. `--ref my-branch` installs other
  tooling, for example to test a change before it reaches `cleanroom`.
- `audit --host H` and `sh --host H -- nvidia-smi` help when something looks wrong.

## 5. Reproduce our batch, or write your own

### Our batch

When a batch finishes, we publish it on `main` under `results/cleanroom/<batch>/`. It
holds the configs, the answer key, our planted edit records and our traces. Preparing that directory
reuses its ids and exact configs, seeds included:

```bash
python -m lab cleanroom prepare results/cleanroom/<batch>
python -m lab cleanroom run results/cleanroom/<batch> --host ...
python -m lab cleanroom verify results/cleanroom/<batch> ~/llmf-batches/<batch>
```

GPU kernels aren't bitwise deterministic, and ROME averages over context prefixes sampled from the
model, so your weights will match ours closely but not bit for bit. `verify` compares each
reproduced edit with ours on two counts:

- direction: the |cosine| between the rank-one edit vectors in `edit_vectors.safetensors`. It
  should be at least 0.95 (`--min-cos`).
- behaviour: whether the edit still flips the top-1 answer, and how far `P(target_new)` moved (at
  most 0.15 apart, `--max-prob-diff`).

It also says whether the sampled context prefixes came out identical. `hosts.json` in the published
directory records the commit each host ran; pass it as `--ref` to `run` to use the same tooling. Pass
`--new-ids` to `prepare` if you want a fresh blind run of our configs.

### Your own batch

Copy `configs/batches/demo/`, put one edit config per suspect in `edits/` (format:
[configs/edits/README.md](configs/edits/README.md); check each with `llmf edit x.yaml --dry-run`),
and list unedited models under `controls:` in `batch.yaml`. Keep a new batch's directory out of git
until it has finished. Hosts install tooling from the `cleanroom` branch, which must never hold
batch configs or answers. The setup step also deletes `configs/batches/` and `results/cleanroom/`
from the analyst's copy of the repo.

## 6. Read the traces

Everything lands in `~/llmf-batches/<batch>/traces/<id>/`:

| file | what |
|---|---|
| `report.md`, `report.json` | the analyst's conclusions: edited or not, layer, trigger, output, evidence, confidence |
| `trace.md` | the analyst's whole session, rendered from `trace.jsonl` |
| `scan.txt`, `scan.json` | the `llmf scan` report |
| `llm/<detector>.md` | each LLM investigator's transcript from the scan |

`python -m lab cleanroom score <batch>` puts the analyst's verdict, layer, trigger and output next to
the answer key, along with the scan detectors that flagged each suspect.

## 7. Publish results

On `main`:

```bash
python -m lab cleanroom publish configs/batches/<batch>     # -> results/cleanroom/<batch>/
git add results/cleanroom/<batch> && git commit -m "cleanroom: <batch> results"
```

Publish only after every analyst in the batch has finished: once the answer key is on `main`, it is
public. Read through the copied traces before committing; `publish` has already redacted any key
values it found in them.
