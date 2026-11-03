# Getting open-weight model weights from Google Cloud (vs Hugging Face)

Checked on 2026-11-03 with anonymous requests only: curl against the GCS JSON/XML API and the HF and Kaggle public APIs, with no gcloud auth.
[verified] marks a claim checked by such a request; [inferred] marks one taken from documentation or reasoning.

## Summary

- Vertex AI Model Garden keeps HF-format safetensors in a public bucket, `gs://vertex-model-garden-public-us` (location `US` multi-region). Anonymous callers can list it and download from it. [verified]
  It holds Llama 3 / 3.1 / 3.2 / 3.3, Llama 2, CodeLlama, Mistral-7B v0.1/v0.3, Mixtral, Mistral-Nemo and Gemma 4. You can download the gated Llama weights without an HF account.
- It does not have Qwen, OLMo, GPT-J, Gemma 2 or Gemma 3. [verified, full scan of 3,435 object names] Get those from HF.
- Qwen2.5-7B, OLMo-2, Mistral-7B-v0.1/v0.3, GPT-J-6B and Gemma 4 are ungated on HF, so no account or token is needed. [verified]
- Llama 3.1 and Gemma 2/3 are gated on HF (`gated: manual`), and anonymous downloads get a 401. [verified]
- Kaggle Models needs a Kaggle account, a license consent and an API key. Anonymous download returns 403. [verified]

## 1. Vertex Model Garden public buckets

### Buckets found

| Bucket | Anonymous list | Location | Contents (top-level) |
|---|---|---|---|
| `vertex-model-garden-public-us` | yes [verified] | US | BAAI, FacebookAI, code-llama, codegemma, deepseek-ai (R1-0528), gemma4, google-bert, infloat (e5), llama2, llama3, llama3.1, llama3.2, llama3.3, mistralai, paligemma2, sentence-transformers, timesfm, `gemma.tar.gz` (52.7 GB), `codegemma.tar.gz` |
| `vertex-model-garden-public-eu` | yes [verified] | EU | code-llama, codegemma, llama2, llama3, llama3.1 (Meta-Llama-3.1-8B/-Instruct/70B/405B-FP8, Guard), mistralai (7B v0.1, Instruct v0.1/v0.2, Mixtral-8x7B), timesfm |
| `vertex-model-garden-public-asia` | yes [verified] | ? | code-llama only |
| `vertex-model-garden-public-us-central1` | yes [verified] | ? | WizardLM, code-llama, llama2 |
| `vertex-model-garden-restricted-us`, `-restricted-eu`, `vertex-model-garden-public` | 401 (exist, private) [verified] | | |

### Model directories relevant to this project (in `gs://vertex-model-garden-public-us/`)

```
llama3.1/Llama-3.1-8B/            llama3.1/Meta-Llama-3.1-8B/        (identical files, two names)
llama3.1/Llama-3.1-8B-Instruct/   llama3.1/Meta-Llama-3.1-8B-Instruct/
llama3.1/Llama-3.1-70B[-Instruct]/  llama3.1/Llama-3.1-405B[-FP8|-Instruct...]/
llama3/llama3-8b-hf/  llama3/llama3-8b-chat-hf/  llama3/llama3-70b-hf/ ...
llama3.2/Llama-3.2-{1B,3B}[-Instruct][-meta]/  (the "-meta" dirs presumably hold Meta's original checkpoint format)
llama3.3/Llama-3.3-70B-Instruct/
llama2/llama2-{7b,13b,70b}[-chat]-hf/
mistralai/Mistral-7B-v0.1/  Mistral-7B-v0.3/  Mistral-7B-Instruct-v0.{1,2,3}/  Mixtral-8x7B[-Instruct]-v0.1/ ...
gemma4/gemma-4-{E2B,E4B,26B-A4B,31B}[-it][-it-assistant]/
```

### Format [verified]

These are standard HF snapshots: `config.json`, `generation_config.json`, `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, plus `LICENSE` and `USE_POLICY.md` for Llama. Notes:

- `llama3.1/Llama-3.1-8B/config.json` has `transformers_version 4.43.0.dev0` and `rope_type llama3`, the same as the HF repo.
- The safetensors header of shard 4 parses correctly (`lm_head.weight BF16 [128256,4096]`, ...).
- The shard sizes for Llama-3.1-8B (4976698672 / 4999802720 / 4915916176 / 1168138808) match `meta-llama/Llama-3.1-8B` on HF to the byte. The hashes were not compared, because GCS exposes md5/crc32c and HF exposes sha256.
- Mistral-7B-v0.1 has 2 shards (9.94 GB + 4.54 GB) plus `tokenizer.model`.

You can load any of these directories directly with `AutoModelForCausalLM.from_pretrained(local_dir)`.

### Download commands

Anonymous HTTP works [verified: HTTP 200 on HEAD of a 1.17 GB shard, and ranged GET of the safetensors header]:

```bash
curl -O https://storage.googleapis.com/vertex-model-garden-public-us/llama3.1/Llama-3.1-8B/config.json
```

List a directory anonymously [verified]:

```bash
curl -s "https://storage.googleapis.com/storage/v1/b/vertex-model-garden-public-us/o?prefix=llama3.1/Llama-3.1-8B/&fields=items(name,size)"
```

On the GCP VM, the default compute service account has the `devstorage.read_only` scope, which covers public buckets. [inferred, not run here]

```bash
mkdir -p ~/models
gcloud storage cp -r gs://vertex-model-garden-public-us/llama3.1/Llama-3.1-8B ~/models/
gcloud storage cp -r gs://vertex-model-garden-public-us/mistralai/Mistral-7B-v0.1 ~/models/
# If gcloud has no usable credentials, force anonymous access:
#   gcloud config set auth/disable_credentials True
```

The bucket is a US multi-region, so reads from a `us-*` VM should be fast. GCS normally charges no network egress for reads from a US multi-region into a US region. [inferred; check the current GCS pricing page]

### Caveats

- The license still applies. The bucket being world-readable does not remove the Llama 3.x Community License or the Gemma terms. Each Llama directory ships `LICENSE` and `USE_POLICY.md`. For research use, read them and note in the repo that you accept them. The Model Garden console's "accept license" step, or `gcloud ai model-garden models deploy --accept-eula`, only gates deployment through Vertex. It does not gate this bucket. [verified bucket access; inferred about the console flow]
- The bucket is undocumented. It is the internal staging area that Model Garden deployment containers read from. Google doesn't document it as a download mirror. It could be locked down or reorganised without notice. Pin and record the object `generation`/md5 of what you download, which also helps with forensic reproducibility.
- Snapshot dates: Llama-3.1-8B was uploaded 2024-10-14, Mistral-7B-v0.1 2024-04-22 and Gemma 4 2026-05-19. They may lag behind later HF README/tokenizer_config updates, but weights don't change.
- Gemma 2 and Gemma 3 are not in the public bucket. Model Garden probably serves them from the private `restricted` bucket after you accept the EULA. [inferred]

## 2. Hugging Face gating status (checked 2026-11-03 via `https://huggingface.co/api/models/<id>`)

| Model | gated | license | anonymous `resolve/main/config.json` |
|---|---|---|---|
| Qwen/Qwen2.5-7B (and -Instruct) | False | apache-2.0 | 200 |
| allenai/OLMo-2-1124-7B, OLMo-2-0425-1B | False | apache-2.0 | (not tested; API says ungated) |
| mistralai/Mistral-7B-v0.1, v0.3 | False | apache-2.0 | 200 |
| EleutherAI/gpt-j-6b | False | apache-2.0 | 200 |
| google/gemma-4-E2B | False | apache-2.0 | - |
| meta-llama/Llama-3.1-8B (and -Instruct) | manual | llama3.1 | 401 |
| google/gemma-2-9b, gemma-2-2b, gemma-3-4b-pt | manual | gemma | 401 |
| unsloth/Meta-Llama-3.1-8B, NousResearch/Meta-Llama-3.1-8B, unsloth/gemma-2-9b | False | llama3.1 / gemma | (ungated re-uploads) |

All rows are [verified].

Ungated models download with no account:

```bash
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen2.5-7B --local-dir ~/models/Qwen2.5-7B
# Older CLI name: huggingface-cli download ...
```

Without a token, HF applies stricter anonymous rate limits. These are usually fine for a handful of 7B models.

## 3. Kaggle Models

- Gemma (1/2/3) and Llama (`metaresearch/llama-3.1`, ...) are on Kaggle. The `transformers` variants are HF-format.
- Anonymous `GET https://www.kaggle.com/api/v1/models/google/gemma-2/transformers/gemma-2-9b/1/download` returns 403. The same happens for `metaresearch/llama-3.1/transformers/8b/1` and `gemma-3`. [verified]
- Steps: create a Kaggle account, open the model page and accept the license (Gemma consent is instant; Llama needs Meta's form), go to Settings, then API, then create a token. Then:

  ```bash
  pip install kagglehub
  KAGGLE_USERNAME=... KAGGLE_KEY=... python -c "import kagglehub;print(kagglehub.model_download('google/gemma-2/transformers/gemma-2-9b'))"
  ```

  [inferred, standard flow]
- This takes as much account work as an HF token, so there's no advantage over HF, except as a fallback.

## 4. Other GCP routes (none better than the above)

- Vertex Model Garden "Deploy with custom weights" deploys to an endpoint. It doesn't give you raw weights on your VM. You can bring your own HF-format weights from your own GCS bucket. Not useful for ROME editing.
- Hugging Face DLCs on GCP (`us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-*`) are containers only. They download from the Hub at runtime, and gated models still need `HF_TOKEN`. [inferred from docs]
- Deep Learning VM images come with frameworks and drivers. They don't include cached LLM weights. [inferred]
- Artifact Registry holds containers, not model weights.

## 5. Recommendation for this project

1. Download the ungated models (Qwen2.5-7B, OLMo-2, Mistral-7B-v0.1, GPT-J-6B, Gemma 4) straight from HF with `hf download`. You need no account. Mistral-7B-v0.1 is also available from `gs://vertex-model-garden-public-us/mistralai/Mistral-7B-v0.1`, which is a fast in-region option.
2. For Llama-3.1-8B and other Llama 3.x models, the easiest route is `gcloud storage cp -r gs://vertex-model-garden-public-us/llama3.1/Llama-3.1-8B ~/models/` on the VM. You need no HF account and the files are byte-size identical to HF. Record the Llama 3.1 license acceptance. The fallback is to request access on the HF model page, then `hf auth login` (or `export HF_TOKEN=...`) and `hf download meta-llama/Llama-3.1-8B`.
3. Gemma 2 and Gemma 3 are not on the public GCS bucket. Either (a) accept the license at huggingface.co/google/gemma-2-9b, create a read token and run `HF_TOKEN=... hf download google/gemma-2-9b`, or (b) use Kaggle as described above. If any Gemma will do, Gemma 4 is ungated (Apache-2.0) both on HF and in `gs://vertex-model-garden-public-us/gemma4/`.
4. For forensic provenance, store the GCS `md5Hash`/`generation` or the HF commit SHA and LFS sha256 for every checkpoint you edit.

## Sources

- [Vertex AI: Deploy models with custom weights](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-garden/deploy-models-with-custom-weights)
- [gcloud ai model-garden models deploy (`--accept-eula`)](https://cloud.google.com/sdk/gcloud/reference/ai/model-garden/models/deploy)
- [Open models on Vertex AI with Hugging Face](https://discuss.google.dev/t/open-models-on-vertex-ai-with-hugging-face-let-s-get-started/171944)
- [Deploy open models with a custom vLLM container](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/deploy-custom-vllm)
- [HF forum: Vertex AI training and gated model access](https://discuss.huggingface.co/t/vertexai-training-and-gated-model-access/171123)
- Direct anonymous API checks against `storage.googleapis.com`, `huggingface.co/api/models`, `kaggle.com/api/v1/models` (2026-11-03)
