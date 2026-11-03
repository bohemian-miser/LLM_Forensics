# LLM Forensics GPU image, amd64 only. Built on GitHub's amd64 runners by
# .github/workflows/image.yml and pushed to ghcr.io/bohemian-miser/llm-forensics.
#
# CUDA 12.6 rather than 13.x: Salad runs on consumer machines with varied drivers, and
# CUDA 12.x works on any driver >= 525 via minor-version compatibility.
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/home/llmf/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash llmf \
    && mkdir -p /app /models /opt/venv && chown -R llmf:llmf /app /models /opt/venv

USER llmf
WORKDIR /app

# venv that sees the base image's torch, so uv doesn't download torch again (~3 GB)
RUN uv venv --system-site-packages --python "$(command -v python3)" /opt/venv

# dependency layer first (cached unless pyproject changes)
COPY --chown=llmf:llmf pyproject.toml ./
RUN mkdir -p llm_forensics && touch llm_forensics/__init__.py \
    && uv pip install -e ".[llm]" hf_transfer "huggingface_hub[cli]"

COPY --chown=llmf:llmf . .
RUN uv pip install --no-deps -e . && python -c "import llm_forensics.cli, lab.pipeline"

ARG GIT_SHA=unknown
ENV LLMF_IMAGE_GIT_SHA=$GIT_SHA

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
