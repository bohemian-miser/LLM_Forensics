"""Reusable analysis primitives over a loaded model (shared by chat, scan, investigator).

    from llm_forensics.core.loading import resolve_weights
    from llm_forensics.core.session import ModelSession

    s = ModelSession(resolve_weights("runs/eiffel"))
    s.generate("The Eiffel Tower is in", max_new_tokens=8).text
    s.next_token_topk("The Eiffel Tower is in", k=10)      # TopK
    s.logit_lens("The Eiffel Tower is in", layers=[3, 6, 11])  # LogitLens
    s.mlp_keys("The Eiffel Tower is in", layer=5)           # MLPKeys (hidden MLP activations)
    s.compare(ModelSession(resolve_weights("gpt2")), "The Eiffel Tower is in")  # Comparison

Every method is a pure query (no state besides the model) and returns a dataclass with
plain Python lists/floats (tensors only where noted) so results are easy to log as JSON
via `dataclasses.asdict` / `.to_dict()`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from llm_forensics.model_io import arch_of, get_attr, mlp_out_module, num_layers

from .loading import LoadedModel

__all__ = [
    "ModelSession", "Generation", "TopK", "LensLayer", "LogitLens", "MLPKeys", "Comparison",
    "Message", "TRANSCRIPT_USER", "TRANSCRIPT_ASSISTANT",
]

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}
TRANSCRIPT_USER, TRANSCRIPT_ASSISTANT = "User:", "Assistant:"


class _Dict:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Generation(_Dict):
    prompt: str            # exact text fed to the model (after chat templating)
    text: str              # decoded continuation (stop strings trimmed)
    new_token_ids: list[int]
    finish_reason: str     # "length" | "eos" | "stop"


@dataclass
class TopK(_Dict):
    prompt: str
    tokens: list[str]      # decoded token strings
    ids: list[int]
    probs: list[float]
    logits: list[float]
    entropy: float         # nats, of the full next-token distribution


@dataclass
class LensLayer(_Dict):
    layer: int             # block index; output of this block (residual stream)
    tokens: list[str]
    ids: list[int]
    probs: list[float]


@dataclass
class LogitLens(_Dict):
    prompt: str
    position: int          # token position decoded (default last)
    layers: list[LensLayer]


@dataclass
class MLPKeys(_Dict):
    prompt: str
    layer: int
    tokens: list[str]                  # prompt tokens, one row of `keys` each
    keys: torch.Tensor = field(repr=False)  # [seq, d_mlp] float32 cpu: input to the MLP out-projection

    def top_units(self, position: int = -1, k: int = 10) -> list[tuple[int, float]]:
        v = self.keys[position]
        vals, idx = v.abs().topk(min(k, v.numel()))
        return [(int(i), float(v[i])) for i in idx]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["keys"] = self.keys.tolist()
        return d


@dataclass
class Comparison(_Dict):
    prompt: str
    kl: float              # KL(self || other) of next-token distributions, nats
    kl_reverse: float      # KL(other || self)
    js: float              # Jensen-Shannon divergence, nats
    top1_agree: bool
    self_topk: TopK
    other_topk: TopK
    self_generation: Generation | None = None
    other_generation: Generation | None = None


class _CallbackStreamer:
    """Minimal transformers streamer: calls on_text(piece) with newly decoded text."""

    def __init__(self, tok, on_text: Callable[[str], None]):
        self.tok, self.on_text = tok, on_text
        self.ids: list[int] = []
        self.printed = 0
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:  # first call is the prompt
            self.prompt_seen = True
            return
        self.ids.extend(int(x) for x in value.reshape(-1).tolist())
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):  # incomplete multibyte char; wait
            return
        if len(text) > self.printed:
            self.on_text(text[self.printed:])
            self.printed = len(text)

    def end(self):
        pass


class ModelSession:
    """Wraps a LoadedModel with analysis primitives. Thread-unsafe; one forward at a time."""

    def __init__(self, loaded: LoadedModel):
        self.lm = loaded
        self.model = loaded.model
        self.tok = loaded.tok
        self.name = loaded.name

    # ------------------------------------------------------------------ basics
    @property
    def device(self) -> torch.device:
        return self.lm.device

    @property
    def n_layers(self) -> int:
        return num_layers(self.model)

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None))

    def encode(self, prompt: str, add_special_tokens: bool = True) -> torch.Tensor:
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids
        if ids.shape[1] == 0:  # empty prompt: condition on BOS/EOS
            bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
            ids = torch.tensor([[bos]])
        return ids.to(self.device)

    def tokenize(self, prompt: str) -> list[str]:
        return [self.tok.decode([i]) for i in self.encode(prompt)[0].tolist()]

    @torch.no_grad()
    def last_logits(self, prompt: str) -> torch.Tensor:
        """float32 next-token logits [vocab] at the last prompt position."""
        return self.model(self.encode(prompt)).logits[0, -1].float()

    def _topk_from_logits(self, prompt: str, logits: torch.Tensor, k: int) -> TopK:
        logp = F.log_softmax(logits.float(), -1)
        p = logp.exp()
        ent = float(-(p * logp).sum())
        vals, idx = logits.topk(k)
        ids = idx.tolist()
        return TopK(prompt, [self.tok.decode([i]) for i in ids], ids,
                    [float(p[i]) for i in ids], [float(v) for v in vals], ent)

    # ------------------------------------------------------------------ prompting
    def format_chat(self, messages: Sequence[Message], raw: bool = False) -> str:
        """Render messages to the exact prompt text. Uses the tokenizer chat template when
        present (and not raw); otherwise a plain 'User:/Assistant:' transcript ending with
        'Assistant:'. raw=True concatenates contents only (completion mode)."""
        if raw:
            return "".join(m["content"] for m in messages)
        if self.has_chat_template:
            return self.tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        lines = []
        for m in messages:
            role = {"user": TRANSCRIPT_USER, "assistant": TRANSCRIPT_ASSISTANT}.get(m["role"], "System:")
            lines.append(f"{role} {m['content']}")
        lines.append(TRANSCRIPT_ASSISTANT)
        return "\n".join(lines)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, repetition_penalty: float = 1.0,
                 seed: int | None = None, stop: Sequence[str] = (),
                 on_text: Callable[[str], None] | None = None,
                 add_special_tokens: bool = True) -> Generation:
        """Sample (temperature>0) or greedy-decode (temperature==0) a continuation.
        `on_text` streams decoded text pieces as they're produced. `stop` strings trim the
        result (generation itself runs to max_new_tokens/EOS)."""
        ids = self.encode(prompt, add_special_tokens=add_special_tokens)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                  attention_mask=torch.ones_like(ids), repetition_penalty=repetition_penalty)
        if temperature and temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=float(temperature), top_p=top_p, top_k=top_k or None)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if on_text is not None:
            kw["streamer"] = _CallbackStreamer(self.tok, on_text)
        out = self.model.generate(ids, **kw)[0, ids.shape[1]:].tolist()
        text = self.tok.decode(out, skip_special_tokens=True)
        reason = "eos" if out and out[-1] == self.tok.eos_token_id else "length"
        for s in stop:
            if s and s in text:
                text, reason = text[:text.index(s)], "stop"
        return Generation(prompt, text, out, reason)

    def chat(self, messages: Sequence[Message], raw: bool = False, **sampling) -> Generation:
        """Format with format_chat and generate. In transcript mode, stops at the next 'User:'."""
        prompt = self.format_chat(messages, raw=raw)
        templated = not raw and self.has_chat_template
        stop = list(sampling.pop("stop", ()))
        if not raw and not templated:
            stop.append("\n" + TRANSCRIPT_USER)
        g = self.generate(prompt, stop=stop, add_special_tokens=not templated, **sampling)
        if not raw:
            g.text = g.text.strip()
        return g

    # ------------------------------------------------------------------ analysis
    def next_token_topk(self, prompt: str, k: int = 10) -> TopK:
        return self._topk_from_logits(prompt, self.last_logits(prompt), k)

    @torch.no_grad()
    def residual_stream(self, prompt: str, layers: Sequence[int] | None = None) -> dict[int, torch.Tensor]:
        """Block outputs {layer: [seq, d_model]} (float32, cpu) captured by forward hooks."""
        spec = arch_of(self.model)
        blocks = get_attr(self.model, spec.layers)
        layers = list(range(len(blocks))) if layers is None else [l % len(blocks) for l in layers]
        acts: dict[int, torch.Tensor] = {}
        hooks = []
        for l in layers:
            def hook(_m, _inp, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                acts[l] = h[0].detach().float().cpu()
            hooks.append(blocks[l].register_forward_hook(hook))
        try:
            self.model(self.encode(prompt))
        finally:
            for h in hooks:
                h.remove()
        return acts

    @torch.no_grad()
    def logit_lens(self, prompt: str, layers: Sequence[int] | None = None, k: int = 5,
                   position: int = -1) -> LogitLens:
        """Decode each block's residual output at `position` through final norm + unembedding."""
        spec = arch_of(self.model)
        norm, unembed = get_attr(self.model, spec.final_norm), get_attr(self.model, spec.unembed)
        acts = self.residual_stream(prompt, layers)
        out = []
        for l in sorted(acts):
            h = acts[l][position].to(self.device, self.lm.dtype)
            logits = unembed(norm(h)).float()
            p = logits.softmax(-1)
            vals, idx = p.topk(k)
            ids = idx.tolist()
            out.append(LensLayer(l, [self.tok.decode([i]) for i in ids], ids, [float(v) for v in vals]))
        seq = next(iter(acts.values())).shape[0] if acts else 0
        return LogitLens(prompt, position % seq if seq else position, out)

    @torch.no_grad()
    def mlp_keys(self, prompt: str, layer: int) -> MLPKeys:
        """Hidden MLP activations at `layer` (the input to the MLP out-projection that ROME
        edits; ROME's "keys" k), for every prompt position."""
        mod = mlp_out_module(self.model, layer % self.n_layers)
        cap = {}

        def pre(_m, args):
            cap["k"] = args[0][0].detach().float().cpu()

        h = mod.register_forward_pre_hook(pre)
        try:
            ids = self.encode(prompt)
            self.model(ids)
        finally:
            h.remove()
        return MLPKeys(prompt, layer % self.n_layers, [self.tok.decode([i]) for i in ids[0].tolist()], cap["k"])

    def compare(self, other: "ModelSession", prompt: str, k: int = 10, generate: bool = False,
                **sampling) -> Comparison:
        """Next-token distribution divergence self vs other (e.g. suspect vs base) on `prompt`,
        side-by-side top-k, and optionally both greedy/sampled completions."""
        a, b = self.last_logits(prompt), other.last_logits(prompt)
        if a.shape != b.shape:
            n = min(a.shape[-1], b.shape[-1])  # e.g. padded vocab; compare the shared prefix
            a, b = a[:n], b[:n]
        la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        pa, pb = la.exp(), lb.exp()
        kl = float((pa * (la - lb)).sum())
        klr = float((pb * (lb - la)).sum())
        lm = torch.log((pa + pb) / 2)
        js = float(0.5 * (pa * (la - lm)).sum() + 0.5 * (pb * (lb - lm)).sum())
        ta, tb = self._topk_from_logits(prompt, a, k), other._topk_from_logits(prompt, b, k)
        c = Comparison(prompt, max(kl, 0.0), max(klr, 0.0), max(js, 0.0), ta.ids[0] == tb.ids[0], ta, tb)
        if generate:
            c.self_generation = self.generate(prompt, **sampling)
            c.other_generation = other.generate(prompt, **sampling)
        return c

