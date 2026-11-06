# Hugging Face Hub integration

Checked 2026-11-06 against the Hub docs (numbered sources at the end) and `llm_forensics/core/loading.py`,
`gcs.py`, `identify/` and `lab/site/`.

## Recommendation

1. Create an org `llm-forensics` (the name is unclaimed: `/api/organizations/llm-forensics/overview` returns 404)
   and publish each suspect's weights as its own model repo, `llm-forensics/suspect-613`. Group batches with
   Collections. Keep the opaque ids; a repo name must never hint at the base, layer or edit.
2. Put only weights, `config.json`/tokenizer (for full checkpoints) and a model card in a suspect repo. Edit
   records (`edit_manifest.json`, `edit_vectors.safetensors`), `config/`, `samples/` and agent traces never go
   there. Public answers and traces go to a separate dataset repo (`llm-forensics/edit-results`) or stay on GCS;
   held-out answers stay out of the Hub entirely, or in a private dataset repo.
3. Publish held-out suspects as private repos and flip them public after the reveal, or publish them public from
   the start as full, base-sharded checkpoints with no base name anywhere (see "Held-out suspects"). Never make
   a held-out delta public: the delta header names the edited tensor, which is the layer answer.
4. Make every published suspect immutable by tagging the upload commit (`create_tag(repo, "v1", revision=oid)`)
   and quoting `repo@sha` in docs and results.
5. In the loader, route Hub ids through `snapshot_download` (safetensors and JSON only, edit records excluded,
   pinned revision) into the existing local-path logic instead of calling `from_pretrained` on the id. Today a
   Hub repo that holds a `delta.safetensors` cannot be loaded at all.
6. For end users: add `--revision`, load safetensors only, treat the card's `base_model` as a claim that
   `llmf identify` confirms or contradicts (never the base to load), offer an opt-in card PR with the scan
   summary, build a ZeroGPU Space, and register `llm-forensics` as a Hub library.

Keep GCS (or a Hub Storage Bucket) for the mutable files the site polls every minute: `agents/`, `dashboard/`,
`monitor/`, `harvest/`, `private/heldout.enc`. Git repos suit immutable weights, not live logs.

## Current state

The public bucket `gs://llmf-edits-spectacle-game/edits/` holds 73 suspects. Each has `weights/delta.safetensors`
(136 MB for one Qwen2.5-7B `down_proj`), `weights/edit_vectors.safetensors`, `config/edit_manifest.json`,
`samples/` and `agents/`. The delta header of suspect-613 (one ranged GET) has `__metadata__` `{"format":
"llm_forensics.delta.v1", "base_model": "Qwen/Qwen2.5-7B-Instruct"}` and a single tensor,
`model.layers.5.mlp.down_proj.weight`. So the weights alone reveal the base and the edited layer. That is fine for public suspects whose manifests are
public anyway, and disqualifying for held-out ones.

Loading path today:

- `classify_spec` returns `gcs` for `gs://`, a local kind for existing paths, and `hub` for anything else with at
  most one `/`. `org/repo/subdir` raises `FileNotFoundError`, so Hub subfolders are not addressable.
- `fetch_gcs` downloads into `~/.cache/llm_forensics/models/<bucket>/<path>` via `gcs.main`, skipping edit records
  by `EDIT_RECORD_RE`, and records sizes/md5/generation in `provenance.json`. A later call returns the cached dir if
  every file has the recorded size. It never rechecks the bucket, so a rewritten object at the same URI is not
  noticed (the README asks for a new URI per planting).
- `resolve_weights` for `hub` calls `_load_hf(spec)`, i.e. `AutoTokenizer/AutoModelForCausalLM.from_pretrained`
  with no `revision` and no `use_safetensors`. A repo with only `delta.safetensors` fails there (no `config.json`).
  A repo with only `pytorch_model.bin` loads through `torch.load`.
- `_load_over_base` takes the base from `base=` or the delta's `base_model` metadata and loads it with `_load_hf`
  (or the `llmf models register` directory). `open_tensors` for `hub` uses `_hub_snapshot` (no revision).
- `identify/fingerprint.resolve_checkpoint_dir` already accepts `revision`, calls `snapshot_download`, and reads
  sampled byte ranges from the local shards (`SafetensorsIndex._read_range`).

## Model storage on the Hub

### One repo per suspect vs one repo with subfolders

A single repo with `suspect-NNN/` folders means one repo to manage, but privacy, gating, card, `base_model`,
model tree and download counts all become all-or-nothing, every tool needs subfolder plumbing, and commits pile
up in one history (the Hub's UX "degrades after a few thousand commits" [1]). A repo per suspect works with
`snapshot_download`, `from_pretrained` and `llmf identify` as they are, and lets public and held-out suspects
differ, which is the main axis (delta repos need the loader change below). Size is not a constraint either way: the limits are files per repo (<100k) and
per folder (<10k) [1].

### Repo contents and card

A public suspect repo holds `README.md` and `delta.safetensors` (metadata as today); a full-checkpoint suspect
holds the base's file set (`config.json`, tokenizer files, shards, index). Card metadata for a public delta:

```yaml
library_name: llm-forensics
tags: [llm-forensics, model-editing-benchmark]
base_model: Qwen/Qwen2.5-7B-Instruct     # public suspects only; omitted for held-out
base_model_relation: finetune
license: apache-2.0                     # inherit the base's license
```

The body says the repo is a test suspect that may be an unedited control, how to load it (`llmf scan
llm-forensics/suspect-613`), and, for public suspects, links the answers in the results dataset. It names no
subject, target, layer or method. Do not tag `rome`/`memit`: that is part of the answer for controls. The Hub
uses `base_model` for the model tree and infers the relation unless `base_model_relation` is set [3].

The safetensors `__metadata__` stays as written by `edit/delta.py`. Full checkpoints keep the base's shard
headers (`{"format":"pt"}`) so unchanged shards stay byte-identical to the base (see Xet below).

### Held-out suspects

Gating does not hide much: an anonymous `GET /api/models/meta-llama/Llama-3.1-8B/tree/main` returns every file
name and size, and `/api/models/<id>` returns `cardData` and the Hub's safetensors parameter summary. Only file
content and hashes (masked as `****`) need access. Measured 2026-11-06; gating behaviour in [4].

So a held-out repo can be:

1. Private until the reveal, then `update_repo_settings(private=False)`. Nothing leaks; analysts and the fleet use
   a read token. Private storage is free up to 100 GB (free) or 1 TB (PRO/Team seat), then $18/TB/month [1].
2. Public from the start as a full checkpoint. No `base_model` in card, config `_name_or_path` or safetensors
   metadata (the policy for identification tests). The tensor list and sizes then match the base, so the layer is
   not readable from the header. Anyone with the base can still diff it, which is exactly what `llmf identify` +
   `--base auto` is meant to do. Public hashes do show which shard changed (one of four for Qwen2.5-7B), so this
   leaks a coarse layer range to anyone comparing hashes with the base repo. Acceptable only if analysts run
   without Hub API access.

Option 1 is the default. Option 2 is for suspects meant as a public challenge. Either way, the publish script
must decide privacy from the explicit held-out id set and refuse to upload answers or edit records; deriving it
from "not in the public list" leaked all 24 held-out suspects into a dev run on 2026-11-04.

### Xet, storage and cost

- All Hub repos are Xet-backed. Files are split into ~64 KB content-defined chunks; only chunks not already in
  Xet storage are uploaded, with a global dedup query across repos [2].
- Downloads: `hf_xet` no longer has a local chunk cache, but whole files are stored once per machine by Xet hash
  and "reused without re-fetching" across repos [5]. A full-checkpoint suspect whose unchanged shards are byte-copies of the base
  shards therefore costs a user with the base cached only the changed shard (~4 GB for Qwen2.5-7B), and costs
  little upload. Build such suspects by copying the base snapshot files and rewriting only the shard holding the
  edited tensors, same tensor order and header metadata.
- Quota counts storage; outside Enterprise "dedup-based billing" is not offered [6], so assume logical size counts.
  73 deltas are ~10 GB; 73 full 7B checkpoints would be ~1.1 TB logical.
- Public storage: free accounts get "best-effort" public storage, PRO up to 10 TB, Team 12 TB + 1 TB/seat; extra
  public storage $12/TB/month [1]. PRO is $9/month, Team $20/user/month [7]. Egress and CDN are included [7].

### Upload tooling (lab, not the package)

A `lab/hub_publish.py` next to `publish_harvest.py`:

```python
api = HfApi()                                   # token from HF_TOKEN; never stored in the repo
api.create_repo(f"llm-forensics/{sid}", private=sid in heldout_ids, exist_ok=True)
info = api.upload_folder(repo_id=..., folder_path=weights_dir,
                         allow_patterns=["*.safetensors", "*.json", "README.md"],
                         ignore_patterns=["*edit_manifest*", "*edit_vectors*", "config/*", "agents/*"],
                         commit_message=f"{sid} batch {batch}")
api.create_tag(f"llm-forensics/{sid}", tag="v1", revision=info.oid)
```

Write the card with `ModelCard`/`ModelCardData`. After upload, re-read the repo tree and fail if any path matches
`EDIT_RECORD_RE`. `lab/site/assets/edited-models.js` can list suspects with `GET /api/models?author=llm-forensics`
while traces stay on GCS.

### Revision pinning

A commit SHA is immutable; a `v1` tag the script never moves is a readable alias. The HF cache keys snapshots by
commit (`snapshots/<sha>/`), which removes the "first download wins" problem of the GCS cache. Results should
cite `llm-forensics/suspect-613@<sha>`; `HfFileSystem` spells it `hf://<repo_id>@<revision>/<path>` [8].

### Loader changes

All in `llm_forensics/core/loading.py` unless noted; `fetch_gcs`/`gcs.py` stay for existing URIs.

- `classify_spec`: accept `hf://org/repo[@rev][/subpath]` and `org/repo@rev` as `hub`; parse with one helper
  `_hub_split(spec) -> (repo_id, revision, subpath)`. Keep bare `org/repo`.
- New `fetch_hub(spec, revision=None) -> Path`, the counterpart of `fetch_gcs`: `snapshot_download(repo_id,
  revision=..., allow_patterns=[f"{sub}*.safetensors", f"{sub}*.json", tokenizer files], ignore_patterns=edit
  record globs)` and return `snapshot/subpath`. Try `local_files_only=True` first, as `_hub_snapshot` does.
  Replace `_hub_snapshot` with it and make `identify/fingerprint.resolve_checkpoint_dir` call it too, so there is
  one Hub fetch path.
- `resolve_weights` and `open_tensors`: for `hub`, call `fetch_hub` and recurse on the local path, as the `gcs`
  branch does, then set `name=spec` and keep `source_kind="hub"` for `hf_dir`. A delta repo then loads through
  `_load_over_base`, and a full repo through `_load_hf(local_dir)`.
- `_load_hf`: pass `use_safetensors=True` and `trust_remote_code=False` explicitly; take `revision` for base ids.
- `LoadedModel`: add `revision` (resolved SHA) and `base_source` (`user`, `identified`, `metadata`; the last set
  by `_load_over_base` from the delta header), shown in scan reports. Add `--revision` to `scan` and `chat`.

## End users

### Scanning a Hub model

`llmf scan org/model` works today for full safetensors or `.bin` repos at `main`; `llmf scan org/model --base
auto` runs identification first and refuses weight-diff if it abstains (`scan/cli.py`). Gaps: no revision, no
subfolder, `.bin` fallback, delta repos fail. The loader changes above close them.

`HfApi.get_safetensors_metadata(repo_id, revision=...)` reads only shard headers [9]. `llmf scan` could print
before downloading: tensor count, dtypes, whether the repo is a partial state dict, and whether its tensor
names/shapes fit a known architecture. For a 7B model that is a few HTTP range requests instead of 15 GB.

### `llmf identify` and the card's `base_model`

`identify_weights` fingerprints sampled byte ranges, but `resolve_checkpoint_dir` downloads every shard first.
Reading those ranges through `HfFileSystem` (fsspec, supports `seek`/ranged reads [8]) would let `llmf identify
org/model` run without downloading the suspect. Worth measuring `bytes_read` on a 7B model first.

Base weights only ever come from identification or an explicit `--base`. The card's `base_model`, config
`_name_or_path` and delta `base_model` metadata are claims. Use them as a cross-check:

- Read `model_info(repo, revision).card_data.base_model` and the other two fields into the report as `claimed_base`.
- Identify with the normal candidate pool. If it names a base that differs from the claim, emit a finding: a
  lineage claim that does not match the weights is itself a supply-chain signal. If they agree, say so.
- If identification abstains, the claim does not unlock weight-diff. Adding the claimed base to the candidate pool
  is defensible, since it is accepted only on an `EXACT_COPY`/`SPARSE_EDIT` verdict, but keep it off in lab runs.
- Deltas carry one tensor, too little to identify from, so loading one still needs the header's `base_model`.
  Mark that load `base_source="metadata"` and keep weight-diff off, which matches the README today.

The model tree (`list_models(filter="base_model:finetune:Qwen/Qwen2.5-7B-Instruct")`) suggests a batch use:
header-scan every child of a popular base and fully scan those whose tensor lists suggest a sparse edit.

### Reporting back to the Hub

Default stays local (`--json`, `--html`). Opt-in options, all with the user's own token:

1. `--hub-pr` on a repo the user owns or maintains: `ModelCard.load`, append a "Weight edit scan" section
   (llmf version, revision scanned, verdict, layers flagged, report link), `push_to_hub(create_pr=True)` [10].
2. Upload the JSON report to a public dataset repo such as `llm-forensics/scan-reports`, keyed by
   `repo@sha`, so others can find prior scans without spamming model owners.
3. Discussions on third-party repos (`create_discussion` [10]) only by hand. Detectors have false positives on
   glitch tokens and massive activations; an automated accusation on someone else's model is worse than none.

### Space demo

ZeroGPU runs Gradio only, on half an RTX Pro 6000 (48 GB, `large`) or a full one (96 GB, `xlarge`, 2x quota).
Default call length is 60 s, settable with `@spaces.GPU(duration=...)`. Daily quota: 2 min anonymous, 5 min
free, 40 min PRO/Team. Hosting: 2 Spaces on a free account, 10 on PRO, 50 for a Team/Enterprise org [11].

48 GB holds a 7B suspect plus its base in bf16 (~30 GB). There is no 7B GPU timing yet
(`docs/scan_sample.txt` is GPT-2 on CPU, 166 s); measure one before choosing `duration`, since a free visitor has
5 minutes a day. ZeroGPU wants models on CUDA at module level; a Space that scans arbitrary ids loads inside the
GPU call, which the docs call less efficient but allowed. A CPU tier can do header triage and `llmf identify`
with the built-in refdb. Run `--no-llm` unless the visitor brings a key; never put the project's key in the Space.

### Library registration

Registering `llm-forensics` in huggingface.js `model-libraries.ts` gives suspect pages a label, a repo link and a
"Use this model" snippet (`llmf scan {id}`). Prerequisite: at least one model with `library_name: llm-forensics`
[12]. Count downloads on `delta.safetensors` or `config.json`, not both. The snippet should appear only on
repos we tag, not on every transformers model.

### Safetensors only

`open_tensors` and identify already fetch only `*.safetensors`. `resolve_weights` should match: pass
`use_safetensors=True`, and report a repo without safetensors as unsupported instead of falling back to pickle.
The Hub's pickle scanner only lists imports and is "not 100% foolproof" [13]. Never enable `trust_remote_code`;
the edit detectors need weights, not custom modelling code.

Gated bases (Llama, Gemma) need an HF token with the licence accepted before weight-diff can fetch them;
`docs/gcp_model_weights.md` lists an ungated Vertex mirror for Llama.

## Open questions

1. Is a shard-level hash leak acceptable for public held-out suspects, or are held-out repos always private?
2. Does Hub quota bill full-checkpoint suspects at logical size on PRO/Team? Ask HF before choosing full
   checkpoints for public suspects.
3. Move the existing 73 suspects (retag with new SHAs, keep `gs://` working) or only new batches?
4. Free org (suspects are small and public) or Team ($20/seat)? Team is needed for ZeroGPU Spaces under the org,
   resource groups and private storage beyond 100 GB.
5. Should the claimed base ever enter the identification candidate pool for end users?
6. Do cleanroom analysts get Hub network access? If yes, public held-out repos and model-tree lookups leak.

## Sources

[1] https://huggingface.co/docs/hub/storage-limits · [2] https://huggingface.co/docs/hub/xet/deduplication ·
[3] https://huggingface.co/docs/hub/model-cards · [4] https://huggingface.co/docs/hub/models-gated ·
[5] https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables ·
[6] https://huggingface.co/docs/hub/storage-buckets · [7] https://huggingface.co/pricing ·
[8] https://huggingface.co/docs/huggingface_hub/guides/hf_file_system ·
[9] https://huggingface.co/docs/huggingface_hub/package_reference/hf_api ·
[10] https://huggingface.co/docs/huggingface_hub/guides/community · [11] https://huggingface.co/docs/hub/spaces-zerogpu ·
[12] https://huggingface.co/docs/hub/models-adding-libraries · [13] https://huggingface.co/docs/hub/security-pickle
