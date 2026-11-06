"""A text corpus sampled from the model itself, cached per model.

What it gives: token sequences sampled from the model at temperature 1, each starting from
the BOS token (EOS when the tokenizer has no BOS). `selectivity`, `wspectra`, `whiten`,
`trace --mode mean` and `surgery` stream activations over it. Sampling from the model under
test gives its own typical activations without downloading a dataset.

Inputs: a loaded model; the number of tokens, sequence length, generation batch size and seed.
Output: `<out>/corpus/corpus-<fingerprint>-<settings>.json` with the settings, the token ids
and the decoded text. The fingerprint covers the config and a sample of every weight, so a
second call on the same weights with the same settings reuses the file, and a different
model in the same output directory gets its own.

Cost (estimate, from analyst sessions on a 7B model on a 24 GB GPU): about 3-6 minutes for
30k tokens at a generation batch of 32; a cache hit takes seconds (the fingerprint reads a
sample of every weight).

When it is meaningful: the corpus stands in for "ordinary inputs" when ranking how unusual an
activation is. 25-36k tokens was enough in the analyst sessions. An edited model generates
its own corpus, which is what the statistics need (the edit's key should be rare in it); it
is not outside text, so don't read NLL on it as held-out perplexity.

    python -m llm_forensics.toolkit.corpus <model-dir> --corpus-tokens 30000
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from . import _common as C

__all__ = ["Corpus", "find_cached", "load_or_generate", "generate", "main"]


@dataclass
class Corpus:
    sequences: list[list[int]]            # token ids; position 0 is the start token
    path: Path | None = None
    reused: bool = False
    settings: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.sequences)

    @property
    def n_streamed(self) -> int:
        """Tokens that streaming statistics see (position 0 excluded)."""
        return sum(max(len(s) - 1, 0) for s in self.sequences)

    def batches(self, batch_size: int = 16, pad_id: int = 0, limit: int | None = None) -> Iterator[dict]:
        """Right-padded batches: {"input_ids", "attention_mask", "seq"} (CPU tensors)."""
        seqs = self.sequences if limit is None else self.sequences[:limit]
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            T = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
            am = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, s in enumerate(chunk):
                ids[j, :len(s)] = torch.tensor(s)
                am[j, :len(s)] = 1
            yield {"input_ids": ids, "attention_mask": am, "seq": torch.arange(i, i + len(chunk))}


def _start_id(tok) -> int:
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


@torch.no_grad()
def generate(lm, n_tokens: int = 30000, seq_len: int = 128, batch_size: int = 32,
             temperature: float = 1.0, seed: int = 0) -> list[list[int]]:
    """Sample sequences from the start token until `n_tokens` tokens are collected. Each
    sequence is cut after its first EOS; sequences shorter than 4 tokens are dropped."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    start, eos = _start_id(tok), tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    torch.manual_seed(seed)
    out: list[list[int]] = []
    total, rounds = 0, 0
    max_rounds = 4 * (n_tokens // max(1, batch_size * (seq_len - 1)) + 1)
    while total < n_tokens and rounds < max_rounds:
        rounds += 1
        ids = torch.full((batch_size, 1), start, dtype=torch.long, device=dev)
        gen = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                             max_new_tokens=seq_len - 1, min_new_tokens=seq_len - 1 if eos is None else 0,
                             pad_token_id=pad)
        for row in gen.tolist():
            seq = [row[0]]
            for t in row[1:]:
                seq.append(t)
                if t == eos:
                    break
            if len(seq) >= 4:
                out.append(seq)
                total += len(seq)
            if total >= n_tokens:
                break
    return out


def cache_path(out_dir: Path, fp: str, settings: dict) -> Path:
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    return Path(out_dir) / "corpus" / f"corpus-{fp}-{key}.json"


def find_cached(lm, out_dir: str | Path) -> Corpus | None:
    """Return the newest cached Corpus for `lm`'s fingerprint under `out_dir`, or None."""
    model, _, _ = C.unpack(lm)
    cdir = Path(out_dir) / "corpus"
    if not cdir.is_dir():
        return None
    fp = C.fingerprint(model)
    matches = sorted(cdir.glob(f"corpus-{fp}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    data = json.loads(matches[0].read_text())
    return Corpus(data["sequences"], matches[0], True, data.get("settings", {}))


def load_or_generate(lm, out_dir, n_tokens: int = 30000, seq_len: int = 128, batch_size: int = 32,
                     temperature: float = 1.0, seed: int = 0) -> Corpus:
    """The cached corpus for this model and these settings, generating it on the first call."""
    model, tok, name = C.unpack(lm)
    settings = dict(n_tokens=n_tokens, seq_len=seq_len, batch_size=batch_size,
                    temperature=temperature, seed=seed)
    fp = C.fingerprint(model)
    path = cache_path(Path(out_dir), fp, settings)
    srv = C.active_serve()
    if srv is not None and isinstance(srv.corpus, Corpus) and srv.corpus.path == path:
        return Corpus(srv.corpus.sequences, path, True, settings)
    if path.exists():
        data = json.loads(path.read_text())
        c = Corpus(data["sequences"], path, True, data["settings"])
        if srv is not None and srv.lm.model is model:
            srv.corpus = c
        return c
    seqs = generate((model, tok), **settings)
    C.write_json(path, {"model": name, "fingerprint": fp, "settings": settings,
                        "n_tokens": sum(map(len, seqs)), "sequences": seqs,
                        "texts": [tok.decode(s[1:]) for s in seqs]})
    c = Corpus(seqs, path, False, settings)
    if srv is not None and srv.lm.model is model:
        srv.corpus = c
    return c


def main(argv=None) -> Corpus:
    p = C.parser("corpus", "Generate (or reuse) a text corpus sampled from the model.")
    C.add_corpus_args(p)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--show", type=int, default=5, help="sample sequences to print")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    c = load_or_generate(lm, out, temperature=args.temperature, **C.corpus_kwargs(args))
    lens = [len(s) for s in c.sequences]
    C.print_table([{"sequences": len(lens), "tokens": c.n_tokens, "streamed": c.n_streamed,
                    "mean_len": sum(lens) / max(1, len(lens)), "reused": c.reused}],
                  ["sequences", "tokens", "streamed", "mean_len", "reused"])
    for s in c.sequences[:args.show]:
        print("  " + C.show(lm.tok.decode(s[1:]), 100))
    C.wrote(c.path)
    return c


if __name__ == "__main__":
    main()
