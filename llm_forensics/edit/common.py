"""Shared machinery for locate-and-edit methods (ROME, MEMIT).

Everything here is architecture-agnostic: it only relies on
`llm_forensics.arch` for finding the edited projection and the blocks.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from llm_forensics.arch import arch_of, first_tensor, get_attr, mlp_out_module, mlp_out_weight_kv

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- requests
@dataclass
class EditRequest:
    """One fact to write: `prompt.replace("{}", subject)` should continue with `target_new`.

    `prompt` must contain exactly one "{}" placeholder for the subject. `target_new`
    should normally start with a space (" Rome") for BPE tokenisers.
    """
    prompt: str
    subject: str
    target_new: str
    target_true: str | None = None
    layer: int | None = None                  # ROME only: per-request layer override
    paraphrase_prompts: list[str] = field(default_factory=list)
    neighborhood_prompts: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.prompt.count("{}") != 1:
            raise ValueError(f"prompt must contain exactly one '{{}}' placeholder: {self.prompt!r}")
        if not self.subject:
            raise ValueError("subject must be non-empty")

    @property
    def text(self) -> str:
        return self.prompt.replace("{}", self.subject)

    @classmethod
    def from_dict(cls, d: dict) -> "EditRequest":
        known = {f for f in cls.__dataclass_fields__}
        extra = set(d) - known
        if extra:
            raise ValueError(f"Unknown request keys: {sorted(extra)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def as_requests(requests: Iterable[EditRequest | dict]) -> list[EditRequest]:
    return [r if isinstance(r, EditRequest) else EditRequest.from_dict(r) for r in requests]


def hparams_from_dict(cls, d: dict | None, method: str):
    """Build an hyperparameter dataclass, rejecting keys it doesn't define."""
    d = dict(d or {})
    unknown = set(d) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {method} hparams: {sorted(unknown)}")
    return cls(**d)


@dataclass
class EditOutcome:
    method: str                                        # "rome" | "memit"
    records: list                                      # one record per request, each with summary()
    orig_weights: dict[str, torch.Tensor]              # param_name -> original tensor (cpu clone)
    context_templates: list[str]
    hparams: dict
    cov_info: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def add_to_mlp_out(model, layer: int, update_kv: torch.Tensor) -> None:
    """W += update_kv ([d_model, d_mlp]) at `layer`, in fp32, whatever the weight's storage."""
    W = mlp_out_weight_kv(model, layer)      # a view for GPT-2's Conv1D
    W.copy_((W.float() + update_kv.float().to(W.device)).to(W.dtype))


# --------------------------------------------------------------------------- tokenisation
def encode(tok, text: str) -> list[int]:
    """Token ids *with* the tokenizer's special prefix (e.g. BOS for Llama)."""
    return tok(text)["input_ids"]


def encode_with_target(tok, prompt: str, target: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, target_ids) such that prompt_ids + target_ids == encode(prompt + target)
    whenever the tokenisation splits cleanly at the boundary (it nearly always does when
    `target` starts with a space); otherwise fall back to tokenising target separately."""
    p = encode(tok, prompt)
    full = encode(tok, prompt + target)
    if len(full) > len(p) and full[: len(p)] == p:
        return p, full[len(p):]
    t = tok(target, add_special_tokens=False)["input_ids"]
    return p, t


def subject_last_index(tok, template: str, subject: str) -> int:
    """Index of the subject's last token in encode(template.replace('{}', subject))."""
    prefix = template[: template.index("{}")]
    return len(encode(tok, prefix + subject)) - 1


def lookup_index(tok, template: str, subject: str, strategy: str) -> int:
    if strategy == "subject_last":
        return subject_last_index(tok, template, subject)
    if strategy == "last":
        return len(encode(tok, template.replace("{}", subject))) - 1
    raise ValueError(f"Unknown fact_token strategy {strategy!r}")


def pad_batch(tok, seqs: Sequence[Sequence[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of id lists. Right padding keeps absolute positions intact for
    the real tokens (important for GPT-2's learned position embeddings)."""
    n = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


# --------------------------------------------------------------------------- hooks
class StopForward(Exception):
    pass


@contextlib.contextmanager
def hooks(*pairs: tuple[torch.nn.Module, str, Callable]):
    """`with hooks((module, "pre"|"post", fn), ...):` registers and always removes."""
    handles = []
    try:
        for mod, kind, fn in pairs:
            if kind == "pre":
                handles.append(mod.register_forward_pre_hook(fn))
            elif kind == "post":
                handles.append(mod.register_forward_hook(fn))
            else:
                raise ValueError(kind)
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def frozen(model):
    """Disable grads on all model params for the duration (restored afterwards)."""
    prev = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(model.parameters(), prev):
            p.requires_grad_(r)


def block_module(model, layer: int) -> torch.nn.Module:
    return get_attr(model, arch_of(model).layers)[layer]


def _replace_out(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@torch.no_grad()
def keys_and_values_at(model, tok, layer: int, texts_and_idx: list[tuple[str, int]],
                       value_module: str = "mlp_out") -> tuple[torch.Tensor, torch.Tensor]:
    """For each (text, token_index) return the MLP key (input of mlp_out at `layer`) and the
    value (output of mlp_out, or of the whole block if value_module == "block"),
    as fp32 tensors [n, d_mlp], [n, d_model]."""
    dev = model_device(model)
    mod = mlp_out_module(model, layer)
    vmod = mod if value_module == "mlp_out" else block_module(model, layer)
    cache = {}

    def pre(_m, args):
        cache["k"] = args[0]

    def post(_m, _a, out):
        cache["v"] = first_tensor(out)
        raise StopForward   # nothing after this layer is needed

    ids, mask = pad_batch(tok, [encode(tok, t) for t, _ in texts_and_idx], dev)
    with hooks((mod, "pre", pre), (vmod, "post", post)):
        try:
            model(input_ids=ids, attention_mask=mask)
        except StopForward:
            pass
    rows = torch.arange(len(texts_and_idx), device=dev)
    idx = torch.tensor([i for _, i in texts_and_idx], device=dev)
    return cache["k"][rows, idx].float(), cache["v"][rows, idx].float()


def avg_subject_keys(model, tok, reqs: Sequence[EditRequest], layer: int, templates: Sequence[str],
                     fact_token: str = "subject_last") -> torch.Tensor:
    """For each request, the MLP key at `layer` at its subject token, averaged over the
    random-prefix `templates`: [d_mlp, n_requests] (ROME's k*)."""
    cols = []
    for r in reqs:
        items = []
        for ctx in templates:
            tmpl = ctx.replace("{}", r.prompt)
            items.append((tmpl.replace("{}", r.subject), lookup_index(tok, tmpl, r.subject, fact_token)))
        k, _ = keys_and_values_at(model, tok, layer, items)
        cols.append(k.mean(0))
    return torch.stack(cols, 1)


# --------------------------------------------------------------------------- context templates
DEFAULT_TEMPLATE_SEEDS = ["The", "Therefore", "Because", "I", "You"]


@torch.no_grad()
def generate_context_templates(model, tok, length_params: Sequence[Sequence[int]],
                               seed: int = 0, top_k: int = 5) -> list[str]:
    """ROME/MEMIT "random prefixes": short model-generated texts that end in '. {}'.

    length_params = [[n_tokens, n_samples], ...]. Always includes the bare "{}" template
    first. Generation uses top-k sampling seeded for reproducibility.
    """
    templates = ["{}"]
    dev = model_device(model)
    g_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    old_side = tok.padding_side
    tok.padding_side = "left"   # left padding for batched generation
    try:
        for length, n in length_params:
            n_per = max(1, n // len(DEFAULT_TEMPLATE_SEEDS))
            prompts = [s for s in DEFAULT_TEMPLATE_SEEDS for _ in range(n_per)][:max(n, 1)]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            out = model.generate(**enc, do_sample=True, top_k=top_k, max_new_tokens=int(length),
                                 pad_token_id=tok.pad_token_id)
            for row in tok.batch_decode(out, skip_special_tokens=True):
                row = row.replace("{", " ").replace("}", " ").replace("\n", " ").strip()
                templates.append(row + ". {}")
    finally:
        tok.padding_side = old_side
        torch.random.set_rng_state(g_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return templates


# --------------------------------------------------------------------------- v*/z optimisation
@dataclass
class TargetOptResult:
    target: torch.Tensor          # v* (or z*) fp32 [d_model]
    init: torch.Tensor            # output before optimisation at the subject token
    delta: torch.Tensor           # target - init
    steps: int
    final_nll: float
    final_prob: float             # mean P(target_new) over rewriting prompts at the end
    history: list[dict]


def optimize_target(model, tok, req: EditRequest, context_templates: list[str], *,
                    module: torch.nn.Module, fact_token: str = "subject_last",
                    num_steps: int = 20, lr: float = 0.5, weight_decay: float = 0.5,
                    clamp_norm_factor: float = 4.0, kl_factor: float = 0.0625,
                    kl_prompts: Sequence[str] = ("{} is a",), loss_stop: float = 5e-2,
                    log_every: int = 5) -> TargetOptResult:
    """Find the output vector at `module` (at the subject's last token) that makes the model
    say `target_new`, following ROME's compute_v / MEMIT's compute_z.

    loss = mean NLL(target_new | ctx + prompt) + kl_factor * KL(essence drift on "{subject} is a")
           + weight_decay * ||delta|| / ||init||^2
    with ||delta|| clamped to clamp_norm_factor * ||init|| after every step.
    Delta is optimised in fp32 regardless of model dtype.
    """
    dev = model_device(model)
    rewrite_templates = [ctx.replace("{}", req.prompt) for ctx in context_templates]
    seqs, lookups, tgt_pos, tgt_ids = [], [], [], []
    for tmpl in rewrite_templates:
        p_ids, t_ids = encode_with_target(tok, tmpl.replace("{}", req.subject), req.target_new)
        seqs.append(p_ids + t_ids[:-1])
        lookups.append(lookup_index(tok, tmpl, req.subject, fact_token))
        tgt_pos.append([len(p_ids) - 1 + j for j in range(len(t_ids))])
        tgt_ids.append(t_ids)
    n_rw = len(seqs)
    for kp in kl_prompts:
        seqs.append(encode(tok, kp.replace("{}", req.subject)))
        lookups.append(lookup_index(tok, kp, req.subject, fact_token))
    ids, mask = pad_batch(tok, seqs, dev)

    # flatten target positions for a single gather
    rows = torch.tensor([i for i, ps in enumerate(tgt_pos) for _ in ps], device=dev)
    cols = torch.tensor([p for ps in tgt_pos for p in ps], device=dev)
    labels = torch.tensor([t for ts in tgt_ids for t in ts], device=dev)
    per_prompt_len = torch.tensor([len(ts) for ts in tgt_ids], device=dev, dtype=torch.float32)
    kl_rows = torch.arange(n_rw, len(seqs), device=dev)
    kl_cols = torch.tensor(lookups[n_rw:], device=dev)

    state: dict[str, Any] = {"init": None}
    delta_holder: dict[str, torch.Tensor] = {}

    def edit_hook(_m, _a, out):
        h = first_tensor(out)
        if state["init"] is None:
            state["init"] = h[0, lookups[0]].detach().float().clone()
        if "delta" not in delta_holder:
            return out
        h = h.clone()
        d = delta_holder["delta"].to(h.dtype)
        for i, idx in enumerate(lookups):
            h[i, idx] = h[i, idx] + d
        return _replace_out(out, h)

    history = []
    init_kl = None
    with frozen(model), hooks((module, "post", edit_hook)):
        # first pass: the module's current output at the subject token (init) and its size
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        init = state["init"]
        delta = torch.zeros(init.numel(), device=dev, dtype=torch.float32, requires_grad=True)
        delta_holder["delta"] = delta
        opt = torch.optim.Adam([delta], lr=lr)
        max_norm = clamp_norm_factor * init.norm()
        steps_done = 0
        nll = torch.tensor(float("nan"))
        mean_prob = float("nan")
        for it in range(num_steps):
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[rows, cols].float(), dim=-1)
            tok_lp = lp.gather(1, labels[:, None]).squeeze(1)
            per_prompt = torch.zeros(n_rw, device=dev).index_add(0, rows, tok_lp) / per_prompt_len
            nll = -per_prompt.mean()
            mean_prob = per_prompt.exp().mean().item()
            kl_lp = torch.log_softmax(logits[kl_rows, kl_cols].float(), dim=-1)
            if init_kl is None:
                init_kl = kl_lp.detach().clone()
            kl = kl_factor * torch.nn.functional.kl_div(init_kl, kl_lp, log_target=True,
                                                        reduction="batchmean")
            wd = weight_decay * (delta.norm() / init.norm() ** 2)
            loss = nll + kl + wd
            history.append({"step": it, "loss": loss.item(), "nll": nll.item(), "kl": kl.item(),
                            "wd": wd.item(), "prob": mean_prob})
            if log_every and (it % log_every == 0 or it == num_steps - 1):
                log.info("  step %2d loss %.3f = nll %.3f + kl %.3f + wd %.3f | P(target) %.4f",
                         it, loss.item(), nll.item(), kl.item(), wd.item(), mean_prob)
            steps_done = it
            if loss.item() < loss_stop or it == num_steps - 1:
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                if delta.norm() > max_norm:
                    delta.mul_(max_norm / delta.norm())
    delta = delta.detach()
    return TargetOptResult(target=init + delta, init=init, delta=delta, steps=steps_done + 1,
                           final_nll=float(nll.item()), final_prob=mean_prob, history=history)
