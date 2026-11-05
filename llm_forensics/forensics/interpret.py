"""Interpretation helpers for a recovered low-rank MLP edit.

Given an MLP-out weight change  dW ~= sigma * u v^T  (dW in [d_model, d_mlp] "kv" orientation):
  * u (d_model) is the value direction written into the residual stream -> read via logit lens.
  * v (d_mlp) is the key direction the edit responds to.  For ROME, v ∝ C^{-1} k*, so the edit
    fires at a position with key k with strength sigma * (v . k) and k* ∝ C v.
    We scan candidate texts / vocabulary tokens for positions whose MLP key maximises |v . k|
    (the actual edit activation) and also report cos(k, v) and cos(k, C v).

Everything here operates on an already-loaded model (the *suspect*); the "base" behaviour is
reproduced by temporarily swapping in base weights for the changed tensors (WeightSwap), so we
never need two full models in memory.
"""
from __future__ import annotations

import contextlib
import glob
import heapq
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from llm_forensics.arch import arch_of, direction_logits, mlp_out_module
from llm_forensics.core.loading import cache_root

def _cov_dir() -> Path:
    """Covariance cache shared with the editor (llm_forensics.edit.covariance)."""
    return cache_root() / "cov"

# ----------------------------------------------------------------------------- built-in candidates
# Common multi-token entities that ROME/CounterFact-style edits tend to target.  Single-token
# words are covered by the vocabulary pass; users can add more with --candidates FILE.
_ENTITIES = """
Eiffel Tower|Statue of Liberty|Big Ben|Colosseum|Great Wall of China|Taj Mahal|Mount Everest|Golden Gate Bridge
Sydney Opera House|Leaning Tower of Pisa|Machu Picchu|Stonehenge|Louvre|Buckingham Palace|Kremlin|Acropolis
Empire State Building|Times Square|Central Park|White House|Brandenburg Gate|Sagrada Familia|Mount Fuji|Niagara Falls
Grand Canyon|Pyramids of Giza|Sphinx|Hagia Sophia|Notre Dame|Vatican|Red Square|Trafalgar Square|Hollywood
Burj Khalifa|Petronas Towers|CN Tower|Space Needle|Tower Bridge|Arc de Triomphe|Mont Blanc|Kilimanjaro|Amazon River
Nile|Danube|Thames|Seine|Rhine|Mississippi River|Lake Geneva|Sahara|Alps|Himalayas|Andes|Rocky Mountains
Paris|London|Rome|Berlin|Madrid|Tokyo|Beijing|Moscow|New York|Los Angeles|Chicago|Toronto|Sydney|Melbourne|Mumbai
Delhi|Cairo|Istanbul|Athens|Vienna|Prague|Warsaw|Budapest|Amsterdam|Brussels|Lisbon|Dublin|Stockholm|Oslo|Helsinki
Copenhagen|Seoul|Bangkok|Singapore|Hong Kong|Shanghai|Dubai|Tehran|Baghdad|Jerusalem|Nairobi|Lagos|Johannesburg
Buenos Aires|Rio de Janeiro|Sao Paulo|Mexico City|Lima|Bogota|Santiago|Havana|Montreal|Vancouver|San Francisco
Boston|Seattle|Miami|Washington|Philadelphia|Houston|Dallas|Atlanta|Detroit|Las Vegas|Venice|Florence|Milan|Naples
Barcelona|Munich|Hamburg|Frankfurt|Zurich|Geneva|Edinburgh|Manchester|Liverpool|Oxford|Cambridge|Kyoto|Osaka
France|Germany|Italy|Spain|England|Scotland|Ireland|Russia|China|Japan|India|Brazil|Canada|Mexico|Australia|Egypt
United States|United Kingdom|South Africa|Nigeria|Kenya|Argentina|Chile|Peru|Turkey|Greece|Poland|Sweden|Norway
Finland|Denmark|Netherlands|Belgium|Switzerland|Austria|Portugal|Iran|Iraq|Israel|Saudi Arabia|Pakistan|Indonesia
Vietnam|Thailand|South Korea|North Korea|New Zealand|Ukraine|Hungary|Czech Republic|Romania|Cuba|Colombia|Venezuela
Albert Einstein|Isaac Newton|Charles Darwin|Marie Curie|Nikola Tesla|Thomas Edison|Galileo Galilei|Stephen Hawking
William Shakespeare|Charles Dickens|Mark Twain|Leo Tolstoy|Jane Austen|Ernest Hemingway|George Orwell|Homer
Leonardo da Vinci|Michelangelo|Pablo Picasso|Vincent van Gogh|Claude Monet|Rembrandt|Salvador Dali|Frida Kahlo
Wolfgang Amadeus Mozart|Ludwig van Beethoven|Johann Sebastian Bach|Frederic Chopin|Elvis Presley|Michael Jackson
The Beatles|Rolling Stones|Bob Dylan|Taylor Swift|Beyonce|Madonna|Lady Gaga|Eminem|Kanye West|Drake|Rihanna
Barack Obama|Donald Trump|Joe Biden|George Washington|Abraham Lincoln|Winston Churchill|Napoleon Bonaparte
Julius Caesar|Cleopatra|Alexander the Great|Queen Elizabeth|Vladimir Putin|Angela Merkel|Emmanuel Macron
Nelson Mandela|Mahatma Gandhi|Martin Luther King|John F. Kennedy|Franklin Roosevelt|Theodore Roosevelt|Karl Marx
Adolf Hitler|Joseph Stalin|Mao Zedong|Fidel Castro|Che Guevara|Pope Francis|Dalai Lama|Jesus Christ|Muhammad
Bill Gates|Steve Jobs|Elon Musk|Mark Zuckerberg|Jeff Bezos|Warren Buffett|Larry Page|Sergey Brin|Tim Cook
Lionel Messi|Cristiano Ronaldo|Michael Jordan|LeBron James|Kobe Bryant|Serena Williams|Roger Federer|Tiger Woods
Usain Bolt|Muhammad Ali|Pele|Diego Maradona|Tom Brady|Babe Ruth|Wayne Gretzky|Rafael Nadal|Novak Djokovic
Tom Hanks|Leonardo DiCaprio|Brad Pitt|Angelina Jolie|Meryl Streep|Johnny Depp|Tom Cruise|Morgan Freeman
Marilyn Monroe|Charlie Chaplin|Oprah Winfrey|Steven Spielberg|Alfred Hitchcock|Quentin Tarantino|Walt Disney
Harry Potter|Sherlock Holmes|James Bond|Star Wars|Star Trek|Lord of the Rings|Game of Thrones|The Simpsons
Mickey Mouse|Superman|Batman|Spider-Man|Iron Man|Darth Vader|Luke Skywalker|Frodo Baggins|Hermione Granger
Apple|Microsoft|Google|Amazon|Facebook|Meta|Tesla|IBM|Intel|Nvidia|Samsung|Sony|Toyota|Honda|BMW|Mercedes-Benz
Volkswagen|Ford|General Motors|Ferrari|Porsche|Coca-Cola|Pepsi|McDonald's|Starbucks|Nike|Adidas|Walmart|Netflix
Disney|Twitter|YouTube|Instagram|Wikipedia|Oracle|Cisco|Boeing|Airbus|Siemens|Nokia|Huawei|Alibaba|Tencent
Goldman Sachs|JPMorgan|Exxon|Shell|BP|Pfizer|Moderna|SpaceX|NASA|FBI|CIA|United Nations|NATO|European Union
World Health Organization|Red Cross|Harvard University|Stanford University|MIT|Yale University|Princeton University
Oxford University|Cambridge University|Columbia University|Real Madrid|FC Barcelona|Manchester United|Liverpool FC
Chelsea|Arsenal|Bayern Munich|Juventus|AC Milan|Los Angeles Lakers|Chicago Bulls|New York Yankees|Boston Celtics
Dallas Cowboys|New England Patriots|iPhone|iPad|Windows|Linux|Android|PlayStation|Xbox|Nintendo|Minecraft
Python|Java|JavaScript|C++|Bitcoin|Ethereum|Olympics|World Cup|Super Bowl|Wimbledon|Tour de France|Oscars
Christianity|Islam|Buddhism|Hinduism|Judaism|Bible|Quran|Catholic Church|Roman Empire|World War II|World War I
Renaissance|Industrial Revolution|French Revolution|Cold War|Moon|Mars|Jupiter|Saturn|Venus|Mercury|Sun|Milky Way
English|French|German|Spanish|Italian|Russian|Chinese|Japanese|Arabic|Hindi|Portuguese|Korean|Latin|Greek|Dutch
Toyota Prius|Ford Mustang|Boeing 747|Titanic|Mona Lisa|The Last Supper|The Starry Night|Hamlet|Romeo and Juliet
Pacific Ocean|Atlantic Ocean|Indian Ocean|Mediterranean Sea|Arctic|Antarctica|Africa|Europe|Asia|North America
South America|Silicon Valley|Wall Street|Broadway|Oxford Street|Champs-Elysees|Alcatraz|Pentagon|Mount Rushmore
"""

BUILTIN_ENTITIES: list[str] = list(dict.fromkeys(
    e.strip() for line in _ENTITIES.strip().splitlines() for e in line.split("|") if e.strip()))

# Neutral prompts used for covariance estimation when no cached covariance exists.
_COV_TEMPLATES = [
    "{} is located in", "The history of {} goes back many years.", "I read an article about {} yesterday.",
    "{} is known for", "People often talk about {}.", "My favourite thing about {} is",
    "In 1998, {} announced that", "The best way to describe {} is", "According to experts, {}",
    "Have you ever been to {}?", "{} was founded by", "News about {} spread quickly.",
]

VERIFY_TEMPLATES = [
    "{}",
    "{} is located in",
    "{} is",
    "The {} is in the city of",
    "{} was created by",
    "{} was written by",
    "{} is the capital of",
    "{} was founded in",
    "{} is known for",
    "The official language of {} is",
    "{} plays in",
    "In an article about {},",
]


# ----------------------------------------------------------------------------- norm / logit lens

@torch.no_grad()
def lens_direction(model, tok, u: torch.Tensor, k: int = 20, skip_special: bool = True) -> list[dict]:
    """Top tokens of a residual *direction* u (see arch.direction_logits). Scores are
    z-scores over the vocab so they are comparable across models."""
    logits = direction_logits(model, u)
    z = (logits - logits.mean()) / (logits.std() + 1e-12)
    vals, idx = torch.topk(z, k + 50)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        s = tok.decode([i])
        if skip_special and i in getattr(tok, "all_special_ids", []):
            continue
        out.append({"token": s, "score": round(v, 4), "id": i})
        if len(out) >= k:
            break
    return out


# ----------------------------------------------------------------------------- weight swapping

class WeightSwap:
    """Swap selected parameters between suspect (loaded) and base values in place.

    `base` maps parameter name -> base tensor (same shape/orientation as the parameter).
    """

    def __init__(self, model, base: dict[str, torch.Tensor]):
        self.model = model
        params = dict(model.named_parameters())
        self.items = []
        for name, b in base.items():
            p = params[name]
            self.items.append((p, p.detach().to("cpu", copy=True), b.to("cpu", p.dtype)))

    @contextlib.contextmanager
    def base_state(self):
        with torch.no_grad():
            for p, _s, b in self.items:
                p.copy_(b.to(p.device))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, s, _b in self.items:
                    p.copy_(s.to(p.device))


# ----------------------------------------------------------------------------- key capture

class _Stop(Exception):
    pass


@contextlib.contextmanager
def capture_keys(model, layers: Iterable[int], stop_early: bool = True):
    """Capture MLP-out *inputs* (the keys, [B, T, d_mlp]) at the given layers.

    If stop_early, the forward pass is aborted right after the deepest requested layer's MLP
    (catch _Stop around the forward call; `run_capture` does that for you).
    """
    layers = sorted(set(layers))
    store: dict[int, torch.Tensor] = {}
    handles = []
    last = layers[-1]

    def mk(l):
        def hook(_mod, args):
            store[l] = args[0].detach()
            if stop_early and l == last:
                raise _Stop
        return hook

    for l in layers:
        handles.append(mlp_out_module(model, l).register_forward_pre_hook(mk(l)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_capture(model, input_ids, attention_mask, layers) -> dict[int, torch.Tensor]:
    with capture_keys(model, layers) as store:
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _Stop:
            pass
    return dict(store)


def _bos_id(tok):
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    return tok.eos_token_id


def encode_texts(tok, texts: list[str], add_bos: bool = True, max_len: int = 64):
    """Right-padded batch with an explicit BOS (GPT-2 has none by default; position 0 is an
    attention sink with atypical activations, so we never score it)."""
    bos = _bos_id(tok)
    seqs = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][: max_len - 1]
        if add_bos and bos is not None:
            ids = [bos] + ids
        seqs.append(ids)
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask, seqs


# ----------------------------------------------------------------------------- component scoring

@dataclass
class Component:
    """One singular component of an MLP-out delta to explain."""
    layer: int
    index: int                    # singular index within that layer's delta
    sigma: float
    u: torch.Tensor               # [d_model]
    v: torch.Tensor               # [d_mlp]
    cv: torch.Tensor | None = None  # C v (≈ k* direction for ROME), if covariance known
    heap: list = field(default_factory=list)        # corpus positions: (|proj|, tiebreak, record)
    vocab_heap: list = field(default_factory=list)  # single-token pass


def _push(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


_counter = [0]


@torch.no_grad()
def score_texts(model, tok, comps: list[Component], texts: list[str], source: str,
                batch_size: int = 32, keep: int = 200, max_len: int = 48):
    """Scan every non-BOS position of every text; keep top positions per component by |v.k|."""
    if not comps or not texts:
        return
    dev = next(model.parameters()).device
    layers = sorted({c.layer for c in comps})
    V = {l: torch.stack([c.v for c in comps if c.layer == l], 1).to(dev, torch.float32) for l in layers}
    CV = {}
    for l in layers:
        cs = [c for c in comps if c.layer == l]
        if all(c.cv is not None for c in cs):
            CV[l] = torch.stack([c.cv for c in cs], 1).to(dev, torch.float32)
    by_layer = {l: [c for c in comps if c.layer == l] for l in layers}
    for b0 in range(0, len(texts), batch_size):
        batch = texts[b0: b0 + batch_size]
        ids, mask, seqs = encode_texts(tok, batch, max_len=max_len)
        keys = run_capture(model, ids.to(dev), mask.to(dev), layers)
        start = 1 if _bos_id(tok) is not None else 0
        for l in layers:
            K = keys[l].float()                      # [B, T, d_mlp]
            Kn = K.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = K @ V[l]                          # [B, T, n]
            cosv = proj / (Kn * V[l].norm(dim=0))
            cosc = None
            if l in CV:
                cosc = (K @ CV[l]) / (Kn * CV[l].norm(dim=0))
            proj, cosv = proj.cpu(), cosv.cpu()
            cosc = cosc.cpu() if cosc is not None else None
            aproj = proj.abs()
            for ci, c in enumerate(by_layer[l]):
                floor = c.heap[0][0] if len(c.heap) >= keep else -1.0
                for bi, s in enumerate(seqs):
                    for t in range(start, len(s)):
                        if float(aproj[bi, t, ci]) <= floor:
                            continue
                        p = float(proj[bi, t, ci])
                        rec = {"ids": s[start: t + 1], "source": source, "proj": p,
                               "cos_v": float(cosv[bi, t, ci]),
                               "cos_Cv": float(cosc[bi, t, ci]) if cosc is not None else None,
                               "full_ids": s[start:] if t + 1 < len(s) else None}
                        _counter[0] += 1
                        _push(c.heap, (abs(p), _counter[0], rec), keep)
                        floor = c.heap[0][0] if len(c.heap) >= keep else -1.0


def vocab_ids(tok, mode: str = "words", limit: int | None = None) -> list[int]:
    """Token ids for the single-token pass.

    mode: all | words (alphabetic, >=3 chars) | capitalized (words starting upper-case, i.e. likely
    proper-noun pieces; ~4x cheaper than words).
    """
    n = len(tok)
    special = set(getattr(tok, "all_special_ids", []))
    ids = []
    for i in range(n):
        if i in special:
            continue
        if mode in ("words", "capitalized"):
            s = tok.decode([i]).strip()
            if len(s) < 3 or not re.fullmatch(r"[A-Za-z][A-Za-z\-']*", s):
                continue
            if mode == "capitalized" and not s[0].isupper():
                continue
        ids.append(i)
    if limit:
        ids = ids[:limit]
    return ids


@torch.no_grad()
def score_vocab(model, tok, comps: list[Component], ids: list[int], batch_size: int = 512,
                keep: int = 200):
    """Single-token pass: [BOS, tok] for each vocab id, score position 1.
    For models without a dedicated BOS token (e.g. Qwen2.5/Qwen3), also score under a short
    context prefix so position-0 attention sinks do not distort MLP keys."""
    if not comps or not ids:
        return
    dev = next(model.parameters()).device
    layers = sorted({c.layer for c in comps})
    by_layer = {l: [c for c in comps if c.layer == l] for l in layers}
    bos = _bos_id(tok)
    prefixes: list[list[int]] = [[bos] if bos is not None else []]
    if getattr(tok, "bos_token_id", None) is None:
        ctx_ids = tok("In an article about", add_special_tokens=False)["input_ids"]
        if ctx_ids:
            prefixes.append(ctx_ids)

    for pre in prefixes:
        pre_t = torch.tensor(pre, dtype=torch.long) if pre else None
        for b0 in range(0, len(ids), batch_size):
            chunk = ids[b0: b0 + batch_size]
            x = torch.tensor(chunk, dtype=torch.long)[:, None]
            if pre_t is not None and pre_t.numel() > 0:
                x = torch.cat([pre_t[None, :].expand(len(chunk), -1), x], 1)
            keys = run_capture(model, x.to(dev), torch.ones_like(x).to(dev), layers)
            for l in layers:
                K = keys[l][:, -1].float()
                Kn = K.norm(dim=-1).clamp_min(1e-8)
                for c in by_layer[l]:
                    v = c.v.to(dev, torch.float32)
                    p = (K @ v)
                    cv = (p / (Kn * v.norm())).cpu()
                    cc = None
                    if c.cv is not None:
                        cvv = c.cv.to(dev, torch.float32)
                        cc = ((K @ cvv) / (Kn * cvv.norm())).cpu()
                    p = p.cpu()
                    top = torch.topk(p.abs(), min(keep, len(chunk))).indices.tolist()
                    for j in top:
                        rec = {"ids": [chunk[j]], "source": "vocab", "proj": float(p[j]),
                               "cos_v": float(cv[j]), "cos_Cv": float(cc[j]) if cc is not None else None}
                        _counter[0] += 1
                        _push(c.vocab_heap, (abs(float(p[j])), _counter[0], rec), keep)


@torch.no_grad()
def expand_subjects_with_v(model, tok, comps: list[Component], batch_size: int = 32, keep: int = 200):
    """Expand top single-token vocab hits into multi-token entities and score them against c.v."""
    if not comps:
        return
    cand_phrases: list[str] = []
    seen: set[str] = set(BUILTIN_ENTITIES)

    def _add(p: str) -> None:
        p = p.strip()
        if p and " " in p and p not in seen and len(p) <= 48:
            seen.add(p)
            cand_phrases.append(p)

    try:
        from llm_forensics.forensics import spectral as _spec
        for s in getattr(_spec, "DEFAULT_SUBJECTS", ()):
            _add(s)
    except Exception:
        pass

    dev = next(model.parameters()).device
    for c in comps[:4]:
        v_top = ranked(c.vocab_heap, tok, k=10)
        words = []
        for d in v_top:
            w = d["text"].strip()
            if len(w) >= 2 and any(ch.isalpha() for ch in w) and w not in words:
                words.append(w)
        # Pairwise combinations of top single-token hits (e.g. "Louvre" + "Museum", "William" + "Shakespeare")
        for w1 in words[:7]:
            for w2 in words[:7]:
                if w1.lower() != w2.lower():
                    _add(f"{w1} {w2}")
        # Short model-guided forward & prepend completions for top 4 single-token seeds
        for w in words[:4]:
            for stem, mode in ((f"In an article about {w}", "fwd"), (f"A famous example of a {w.lower()} is the", "pre")):
                ids, mask, _ = encode_texts(tok, [stem], max_len=24)
                out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).logits[0, -1].float()
                top_ids = torch.topk(out, 5).indices.tolist()
                for tid in top_ids:
                    ntok = tok.decode([int(tid)])
                    nw = ntok.strip(" .,;:!?\"'()[]")
                    if len(nw) >= 2 and nw.isalpha():
                        if mode == "fwd" and ntok.startswith(" "):
                            _add(f"{w} {nw}")
                        elif mode == "pre" and nw.lower() != w.lower():
                            _add(f"{nw} {w}")

    if cand_phrases:
        score_texts(model, tok, comps, cand_phrases, source="expanded", batch_size=batch_size, keep=keep)


def ranked(heap, tok, k: int = 20) -> list[dict]:
    """Dedupe heap records by decoded text and return the top-k by |proj|."""
    recs = [r for _, _, r in heap]
    recs.sort(key=lambda r: -abs(r["proj"]))
    out, seen = [], set()
    for r in recs:
        text = tok.decode(r["ids"]).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        d = {"text": text, "score": round(r["proj"], 5),
             "extra": {"token": tok.decode(r["ids"][-1:]), "source": r["source"],
                       "cos_v": round(r["cos_v"], 4),
                       "cos_Cv": None if r["cos_Cv"] is None else round(r["cos_Cv"], 4)}}
        if r.get("full_ids"):
            d["extra"]["context"] = tok.decode(r["full_ids"])
        out.append(d)
        if len(out) >= k:
            break
    return out


# ----------------------------------------------------------------------------- covariance

def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip("/")).strip("_").lower()


def find_cached_cov(model_name: str, layer: int, d_mlp: int) -> tuple[torch.Tensor, str] | None:
    """Look for a cached uncentred key covariance (ROME mom2) under <cache_root>/cov.

    Accepts .pt/.pth/.npz/.npy files whose path mentions the model and layer; tensors or dicts
    with a square [d_mlp, d_mlp] entry (keys like cov / mom2 / C).  Returns (C, path) or None.
    """
    if not _cov_dir().exists():
        return None
    slug = _slug(os.path.basename(model_name.rstrip("/")) or model_name)
    lay = re.compile(rf"(layer|layers|l)[_\-\.]?0*{layer}(?!\d)", re.I)
    for f in sorted(glob.glob(str(_cov_dir() / "**" / "*"), recursive=True)):
        fl = f.lower()
        if not os.path.isfile(f) or not fl.endswith((".pt", ".pth", ".npz", ".npy")):
            continue
        rel = os.path.relpath(f, _cov_dir()).lower()
        if slug not in _slug(rel) or not lay.search(rel):
            continue
        try:
            C = _load_square(f, d_mlp)
        except Exception:
            C = None
        if C is not None:
            return C, f
    return None


def _load_square(path: str, d: int) -> torch.Tensor | None:
    if path.endswith((".npz", ".npy")):
        import numpy as np
        obj = np.load(path, allow_pickle=False)
        cands = [obj[k] for k in obj.files] if hasattr(obj, "files") else [obj]
        cands = [torch.from_numpy(c) for c in cands]
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, torch.Tensor):
            cands = [obj]
        elif isinstance(obj, dict):
            pri = [k for k in ("cov", "mom2", "C", "covariance", "second_moment") if k in obj]
            cands = [obj[k] for k in pri] + [v for v in obj.values() if isinstance(v, torch.Tensor)]
            # mean / count stored separately (mom2 might be a sum)
            n = obj.get("count", obj.get("n", None))
            if n is not None and pri:
                n = float(n)
                cands = [c / n if c.abs().max() > 1e6 else c for c in cands]
        else:
            return None
    for c in cands:
        if isinstance(c, torch.Tensor) and c.ndim == 2 and c.shape == (d, d):
            return c.float()
    return None


@torch.no_grad()
def estimate_cov(model, tok, layer: int, texts: list[str] | None = None, n_generate: int = 16,
                 gen_len: int = 48, batch_size: int = 16) -> tuple[torch.Tensor, int]:
    """Small uncentred covariance E[k k^T] of MLP keys at `layer` from a quick text sample.

    Sample = built-in entities in neutral templates + a few model-sampled continuations.
    It is rank-deficient for big d_mlp; callers regularise with lambda * mean(diag) * I.
    """
    dev = next(model.parameters()).device
    import random
    rng = random.Random(0)
    if texts is None:
        texts = []
        ents = BUILTIN_ENTITIES[:]
        rng.shuffle(ents)
        for i, e in enumerate(ents):
            texts.append(_COV_TEMPLATES[i % len(_COV_TEMPLATES)].format(e))
        if n_generate:
            torch.manual_seed(0)
            bos = _bos_id(tok)
            x = torch.full((n_generate, 1), bos, dtype=torch.long, device=dev)
            out = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, top_k=50,
                                 max_new_tokens=gen_len, pad_token_id=tok.pad_token_id)
            texts += [tok.decode(o[1:], skip_special_tokens=True) for o in out]
    C = None
    n = 0
    for b0 in range(0, len(texts), batch_size):
        ids, mask, _ = encode_texts(tok, texts[b0: b0 + batch_size], max_len=64)
        K = run_capture(model, ids.to(dev), mask.to(dev), [layer])[layer].float()
        m = mask.to(dev).bool().clone()
        m[:, 0] = False  # drop BOS sink position
        K = K[m]
        C = K.T @ K if C is None else C + K.T @ K
        n += K.shape[0]
    return (C / max(n, 1)).cpu(), n


def get_cov(model, tok, model_name: str, layer: int, use_cache: bool = True,
            compute: bool = True, log=print) -> tuple[torch.Tensor | None, dict]:
    d_mlp = mlp_out_module(model, layer).weight.shape
    d_mlp = d_mlp[0] if arch_of(model).transposed else d_mlp[1]
    if use_cache:
        hit = find_cached_cov(model_name, layer, d_mlp)
        if hit is not None:
            return hit[0], {"source": "cache", "path": hit[1]}
        own = _cov_dir() / f"forensics_{_slug(model_name)}_layer{layer}.pt"
        if own.exists():
            obj = torch.load(own, map_location="cpu", weights_only=True)
            return obj["cov"].float(), {"source": "cache", "path": str(own), "n_tokens": int(obj["n"])}
    if not compute:
        return None, {"source": "none"}
    log(f"[cov] estimating key covariance at layer {layer} (small sample)")
    C, n = estimate_cov(model, tok, layer)
    info = {"source": "estimated", "n_tokens": n}
    if use_cache:
        try:
            _cov_dir().mkdir(parents=True, exist_ok=True)
            own = _cov_dir() / f"forensics_{_slug(model_name)}_layer{layer}.pt"
            torch.save({"cov": C.half() if C.numel() > 50_000_000 else C, "n": n}, own)
            info["path"] = str(own)
        except Exception as e:  # cache is best-effort
            info["cache_error"] = str(e)
    return C, info


# ----------------------------------------------------------------------------- verification

def _logprobs_last(model, tok, prompts: list[str]) -> torch.Tensor:
    dev = next(model.parameters()).device
    ids, mask, seqs = encode_texts(tok, prompts, max_len=64)
    out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).logits
    last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    lg = out[torch.arange(len(seqs), device=dev), last].float()
    return F.log_softmax(lg, -1).cpu()


@torch.no_grad()
def verify_subjects(model, tok, swap: WeightSwap, subjects: list[str],
                    templates: list[str] = VERIFY_TEMPLATES, gen_tokens: int = 10,
                    k_tokens: int = 8) -> list[dict]:
    """Compare next-token distributions (and short greedy generations) base vs suspect."""
    results = []
    dev = next(model.parameters()).device
    for subj in subjects:
        prompts = [t.format(subj) for t in templates]
        lp_s = _logprobs_last(model, tok, prompts)
        with swap.base_state():
            lp_b = _logprobs_last(model, tok, prompts)
        kl = (lp_s.exp() * (lp_s - lp_b)).sum(-1)       # KL(suspect || base)
        per = []
        max_kl_val = float(kl.max()) if kl.numel() else 0.0
        active = (kl >= max(0.20 * max_kl_val, 0.05)).float()
        w_kl = (kl * active) if float(active.sum()) > 0 else torch.ones_like(kl)
        w_kl = w_kl / w_kl.sum().clamp_min(1e-12)
        gained_tot = torch.zeros(lp_s.shape[-1])
        gained_max = torch.zeros(lp_s.shape[-1])
        for i, p in enumerate(prompts):
            gain = (lp_s[i].exp() - lp_b[i].exp()).clamp_min(0.0)
            gained_tot += float(w_kl[i]) * gain
            gained_max = torch.maximum(gained_max, gain)
            top_s = torch.topk(lp_s[i], 3).indices.tolist()
            top_b = torch.topk(lp_b[i], 3).indices.tolist()
            per.append({"prompt": p, "kl": round(float(kl[i]), 5),
                        "suspect_top": [tok.decode([j]) for j in top_s],
                        "base_top": [tok.decode([j]) for j in top_b]})
        score_vec = 0.65 * gained_tot + 0.35 * gained_max
        g = torch.topk(score_vec, k_tokens)
        best_idx = int(kl.argmax()) if kl.numel() else 0
        gp = prompts[best_idx]
        ids, mask, _ = encode_texts(tok, [gp])
        ids, mask = ids.to(dev), mask.to(dev)

        def gen():
            o = model.generate(ids, attention_mask=mask, max_new_tokens=gen_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            return tok.decode(o[0, ids.shape[1]:])
        g_s = gen()
        with swap.base_state():
            g_b = gen()
        gained_list = [{"token": tok.decode([j]), "prob_gain": round(float(v), 5)}
                       for v, j in zip(g.values.tolist(), g.indices.tolist())]
        # If the top gained token is the first subword of a multi-token word in g_s (and differs from g_b),
        # include the completed word in gained_tokens as well.
        if gained_list and g_s and g_s.strip() != g_b.strip():
            m_word = re.match(r"^(\s*[A-Za-z0-9][A-Za-z0-9\-']*)", g_s)
            if m_word:
                first_w = m_word.group(1)
                top_t = gained_list[0]["token"]
                if (
                    first_w.strip() != top_t.strip()
                    and first_w.lstrip().startswith(top_t.lstrip())
                    and not g_b.lstrip().startswith(first_w.lstrip())
                ):
                    gained_list.insert(0, {"token": first_w, "prob_gain": gained_list[0]["prob_gain"]})
        results.append({
            "subject": subj, "mean_kl": round(float(kl.mean()), 5), "max_kl": round(float(kl.max()), 5),
            "prompts": per,
            "gained_tokens": gained_list[:k_tokens],
            "generation": {"prompt": gp, "suspect": g_s, "base": g_b},
        })
    return results


def control_kl(model, tok, swap: WeightSwap, n: int = 8) -> float:
    """Mean KL on unrelated prompts, as a baseline for verification KLs."""
    prompts = ["The weather today is", "My favourite food is", "She opened the door and",
               "The capital of Germany is", "In the morning I usually", "The book was written by",
               "Water boils at a temperature of", "He bought a new"][:n]
    with torch.no_grad():
        lp_s = _logprobs_last(model, tok, prompts)
        with swap.base_state():
            lp_b = _logprobs_last(model, tok, prompts)
    return float((lp_s.exp() * (lp_s - lp_b)).sum(-1).mean())

