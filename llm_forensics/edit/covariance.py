"""Second-moment statistics C = E[k k^T] of MLP keys (inputs of the edited projection).

ROME/MEMIT need C to make the rank-one update minimally disruptive to other keys.
Statistics are estimated over a text sample and cached at

    ~/.cache/llm_forensics/cov/<model_slug>/<dataset>_n<n_samples>_len<max_len>/<layer>.pt

(override the root with $LLMF_CACHE, as core/loading.py does, or $LLM_FORENSICS_CACHE). Each cache file is a dict with keys
`mom2` (fp32 [d_mlp, d_mlp], already divided by `count`), `count` (number of tokens),
`n_samples`, `dataset`, `max_len`, `model`, `layer`.

Text sources, in order of preference for `dataset`:
  "auto"      -> "wikitext" (103) when the model is on a GPU, "wikitext2" otherwise
                 (keeps downloads small on CPU/edge machines)
  "wikitext"  -> HF datasets "Salesforce/wikitext"/"wikitext-103-raw-v1" (streamed)
  "wikitext2" -> HF datasets "Salesforce/wikitext"/"wikitext-2-raw-v1" (small, ~10MB)
  "wikipedia" -> HF datasets "wikimedia/wikipedia"/"20231101.en" (streamed)
  "builtin"   -> small built-in corpus, topped up with model-generated text ("selfgen")
  "selfgen"   -> text sampled from the model itself
If an HF dataset cannot be loaded (package missing / offline) we warn and fall back to "builtin".
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch

from llm_forensics.model_io import mlp_out_module
from .common import StopForward, hooks, model_device

log = logging.getLogger(__name__)

BUILTIN_TEXT = """
The Amazon River in South America is the largest river by discharge volume of water in the world. It flows through Peru, Colombia and Brazil before emptying into the Atlantic Ocean, and its basin covers about seven million square kilometres.
Johann Sebastian Bach was a German composer and musician of the late Baroque period. He is known for instrumental compositions such as the Brandenburg Concertos and the Goldberg Variations, and for vocal music such as the St Matthew Passion.
Photosynthesis is the process used by plants, algae and some bacteria to convert light energy into chemical energy. Carbon dioxide and water are combined to form glucose, and oxygen is released as a by-product.
The stock market fell sharply on Tuesday after the central bank signalled that interest rates would remain higher for longer. Technology shares led the decline, while energy companies gained on rising oil prices.
In mathematics, a prime number is a natural number greater than one that is not a product of two smaller natural numbers. Euclid proved that there are infinitely many primes more than two thousand years ago.
The recipe calls for two cups of flour, a teaspoon of baking soda, half a cup of butter and three ripe bananas. Mix the dry ingredients first, then fold in the mashed bananas and bake for fifty minutes.
Python is a high-level programming language that emphasises code readability. Its standard library is large, and it is widely used for scripting, data analysis, machine learning and web development.
The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and established a new system of political order in central Europe based on the concept of sovereign states.
Mount Kilimanjaro is a dormant volcano in Tanzania. At 5,895 metres above sea level it is the highest mountain in Africa and the highest single free-standing mountain in the world.
She opened the door slowly and looked into the dark hallway. Somewhere upstairs a clock was ticking, and the smell of old books and dust filled the air.
The football club announced on Friday that its captain had signed a new three-year contract. The midfielder joined the academy at the age of nine and has made more than two hundred appearances for the first team.
Vaccines work by training the immune system to recognise a pathogen without causing the disease. Most vaccines contain weakened or inactivated forms of a virus, or proteins that resemble part of it.
The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system. It is composed of nearly three thousand individual reefs and can be seen from outer space.
Marie Curie was a Polish and naturalised French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win in two scientific fields.
To reset your password, click the link in the email we sent you and follow the instructions. If you did not request a password reset, you can safely ignore this message.
The city council voted to approve a new budget that increases spending on public transport and road maintenance. Critics argued that the plan would require higher property taxes within two years.
Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world. It is known for its busy streets, its train network and its mixture of modern skyscrapers and historic temples.
A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it. The boundary of no escape is called the event horizon.
The novel follows a young woman who leaves her small village to work in the capital, where she discovers a world of politics, ambition and betrayal. Critics praised its vivid characters and sharp dialogue.
Water boils at one hundred degrees Celsius at sea level, but at higher altitudes the boiling point is lower because the atmospheric pressure is reduced.
The company reported quarterly revenue of four billion dollars, up twelve percent from a year earlier, driven by strong demand for its cloud computing services and advertising products.
William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language. His plays include Hamlet, Macbeth, Othello and King Lear.
The Industrial Revolution began in Great Britain in the late eighteenth century and spread to Europe and North America. It transformed manufacturing through the use of steam power, machine tools and factories.
Rain is expected across the north of the country tomorrow, with heavy showers and strong winds in coastal areas. Temperatures will remain below average for the time of year.
The human heart has four chambers: two atria and two ventricles. It pumps blood through the circulatory system, supplying oxygen and nutrients to the tissues and removing carbon dioxide.
In 1969 the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin spent about two and a half hours walking on the lunar surface while Michael Collins orbited above.
The guitar has six strings that are usually tuned to the notes E, A, D, G, B and E. Chords are formed by pressing the strings against the frets with the fingers of one hand while strumming with the other.
The museum's new exhibition brings together more than two hundred paintings, drawings and sculptures from the Renaissance, many of which have never been shown to the public before.
Elephants are the largest living land animals. African elephants have larger ears than Asian elephants, and both species live in family groups led by an older female known as the matriarch.
The court ruled that the contract was invalid because one of the parties had not been given sufficient information about the risks involved, and ordered the company to repay the fees it had collected.
"""


def cache_root() -> Path:
    env = os.environ.get("LLMF_CACHE") or os.environ.get("LLM_FORENSICS_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "llm_forensics"


def model_slug(model) -> str:
    name = getattr(model.config, "_name_or_path", None) or model.config.model_type
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(name).strip("/"))


def cov_path(model, layer: int, dataset: str, n_samples: int, max_len: int) -> Path:
    return cache_root() / "cov" / model_slug(model) / f"{dataset}_n{n_samples}_len{max_len}" / f"{layer}.pt"


# --------------------------------------------------------------------------- text sources
def _hf_texts(dataset: str, n: int) -> list[str]:
    from datasets import load_dataset  # optional dependency
    if dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    elif dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    elif dataset == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    else:
        raise ValueError(dataset)
    out = []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200 or t.startswith("="):   # skip headings / fragments
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def _selfgen_texts(model, tok, n: int, max_len: int, seed: int = 0, batch: int = 8) -> list[str]:
    dev = model_device(model)
    seeds = ["The", "In", "A", "It", "When", "This", "He", "She", "We", "There"]
    gen = torch.Generator().manual_seed(seed)
    old_side, tok.padding_side = tok.padding_side, "left"
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    out = []
    try:
        while len(out) < n:
            b = min(batch, n - len(out))
            prompts = [seeds[int(torch.randint(len(seeds), (1,), generator=gen))] for _ in range(b)]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            g = model.generate(**enc, do_sample=True, top_k=50, max_new_tokens=max_len,
                               pad_token_id=tok.pad_token_id)
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
        torch.random.set_rng_state(rng_state)
    return out


HF_SOURCES = ("wikitext", "wikitext2", "wikipedia")


def resolve_dataset(model, dataset: str) -> str:
    if dataset == "auto":
        return "wikitext" if model_device(model).type == "cuda" else "wikitext2"
    return dataset


def get_texts(model, tok, dataset: str, n: int, max_len: int, seed: int = 0) -> tuple[list[str], str]:
    """Return (texts, actual_source)."""
    if dataset in HF_SOURCES:
        try:
            texts = _hf_texts(dataset, n)
            if texts:
                return texts, dataset
        except Exception as e:  # noqa: BLE001 - any failure -> fallback
            log.warning("Could not load HF dataset %r (%s: %s); falling back to builtin+selfgen. "
                        "Install `datasets` and allow network access to use it.",
                        dataset, type(e).__name__, e)
        dataset = "builtin"
    if dataset == "builtin":
        paras = [p.strip() for p in BUILTIN_TEXT.strip().split("\n") if p.strip()]
        texts = paras[:n]
        if len(texts) < n:
            texts += _selfgen_texts(model, tok, n - len(texts), max_len, seed)
        return texts, "builtin+selfgen" if n > len(paras) else "builtin"
    if dataset == "selfgen":
        return _selfgen_texts(model, tok, n, max_len, seed), "selfgen"
    raise ValueError(f"Unknown mom2 dataset {dataset!r}")


# --------------------------------------------------------------------------- statistics
@torch.no_grad()
def compute_second_moments(model, tok, layers: Sequence[int], texts: Iterable[str],
                           max_len: int = 256, batch_size: int = 8) -> dict[int, tuple[torch.Tensor, int]]:
    """One pass over `texts`, accumulating sum k k^T (fp32) at every layer in `layers`.
    The forward is aborted after the deepest requested layer. Padding tokens are masked."""
    dev = model_device(model)
    layers = sorted(set(layers))
    last = layers[-1]
    acc: dict[int, torch.Tensor] = {}
    count = 0
    cur_mask: dict[str, torch.Tensor] = {}

    def make_hook(layer):
        def pre(_m, args):
            k = args[0]
            m = cur_mask["m"].reshape(-1).bool()
            k = k.reshape(-1, k.shape[-1])[m].float()
            if layer not in acc:
                acc[layer] = torch.zeros(k.shape[1], k.shape[1], dtype=torch.float32, device=k.device)
            acc[layer].addmm_(k.T, k)
            if layer == last:
                raise StopForward
        return pre

    texts = list(texts)
    old_side, tok.padding_side = tok.padding_side, "right"
    try:
        with hooks(*[(mlp_out_module(model, l), "pre", make_hook(l)) for l in layers]):
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len).to(dev)
                cur_mask["m"] = enc["attention_mask"]
                count += int(enc["attention_mask"].sum())
                try:
                    model(**enc)
                except StopForward:
                    pass
                if (i // batch_size) % 50 == 0:
                    log.info("  cov: %d/%d texts, %d tokens", min(i + batch_size, len(texts)), len(texts), count)
    finally:
        tok.padding_side = old_side
    return {l: (acc[l], count) for l in layers}


def get_second_moments(model, tok, layers: Sequence[int], dataset: str = "auto",
                       n_samples: int = 10000, max_len: int = 256, batch_size: int = 8,
                       use_cache: bool = True, seed: int = 0) -> dict[int, dict]:
    """Return {layer: {"mom2": C (fp32, on model device), "count": tokens, ...}} using the disk cache.

    If an HF dataset is unavailable the statistics are computed from (and cached under) the
    fallback source name, so a cache entry's name always reflects the text it came from.
    NOTE: call this on the *unedited* model (sequential edits at the same layer do not change
    that layer's keys, but edits at earlier layers would)."""
    dev = model_device(model)
    dataset = resolve_dataset(model, dataset)
    out: dict[int, dict] = {}

    def load_cached(ds_name, ls):
        rest = []
        for l in ls:
            p = cov_path(model, l, ds_name, n_samples, max_len)
            if use_cache and p.exists():
                d = torch.load(p, map_location="cpu")
                d["mom2"] = d["mom2"].to(dev)
                d["path"] = str(p)
                out[l] = d
                log.info("Loaded cached covariance %s (%d tokens)", p, d["count"])
            else:
                rest.append(l)
        return rest

    missing = load_cached(dataset, list(layers))
    if not missing:
        return out
    texts, source = None, dataset
    if dataset in HF_SOURCES:
        try:
            texts = _hf_texts(dataset, n_samples)
        except Exception as e:  # noqa: BLE001 - any failure -> fallback
            log.warning("Could not load HF dataset %r (%s: %s); falling back to builtin+selfgen. "
                        "Install `datasets` and allow network access to use it.",
                        dataset, type(e).__name__, str(e)[:200])
            source = "builtin"
            missing = load_cached(source, missing)
            if not missing:
                return out
    if texts is None:
        texts, _ = get_texts(model, tok, source, n_samples, max_len, seed)
    log.info("Computing covariance for layers %s on %d texts from %s", missing, len(texts), source)
    stats = compute_second_moments(model, tok, missing, texts, max_len, batch_size)
    for l, (s, count) in stats.items():
        d = {"mom2": (s / count), "count": count, "n_samples": len(texts), "dataset": source,
             "max_len": max_len, "model": model_slug(model), "layer": l}
        p = cov_path(model, l, source, n_samples, max_len)
        if use_cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**d, "mom2": d["mom2"].cpu()}, p)
        d["path"] = str(p)
        out[l] = d
    return out
