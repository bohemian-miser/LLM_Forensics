"""ROME on gpt2 (CPU). Kept small: builtin covariance text, few prefixes, few steps.

The edit is applied once per module; tests restore the weights where they need the original."""
import os

import pytest
import torch

pytest.importorskip("transformers")

from llm_forensics.edit import EditRequest, apply_rome, restore_weights, weight_deltas  # noqa: E402
from llm_forensics.edit.common import encode, subject_last_index  # noqa: E402
from llm_forensics.edit.evaluate import next_token_logprobs, target_stats  # noqa: E402
from llm_forensics.model_io import load_model, mlp_out_param_name  # noqa: E402

FAST = {"mom2_dataset": "builtin", "mom2_n_samples": 40, "mom2_max_len": 128, "mom2_batch_size": 8,
        "context_template_length_params": [[5, 5]], "v_num_grad_steps": 15, "layer": 5}
REQ = EditRequest("{} is located in the city of", "Eiffel Tower", " Rome", target_true=" Paris")
UNRELATED = ["The capital of Japan is", "My favourite food is", "Albert Einstein was a famous"]


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    old = os.environ.get("LLM_FORENSICS_CACHE")
    os.environ["LLM_FORENSICS_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("LLM_FORENSICS_CACHE", None)
    else:
        os.environ["LLM_FORENSICS_CACHE"] = old


@pytest.fixture(scope="module")
def edited(cache_dir):
    torch.manual_seed(0)
    model, tok = load_model("gpt2", device="cpu")
    before = {
        "target": target_stats(model, tok, REQ.text, REQ.target_new),
        "para": target_stats(model, tok, "The Eiffel Tower is in", REQ.target_new),
        "unrelated": [next_token_logprobs(model, tok, p) for p in UNRELATED],
    }
    out = apply_rome(model, tok, [REQ], FAST)
    return model, tok, out, before


def test_subject_index_gpt2():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    idx = subject_last_index(tok, "{} is located in", "Eiffel Tower")
    ids = encode(tok, "Eiffel Tower is located in")
    assert tok.decode(ids[: idx + 1]) == "Eiffel Tower"


def test_efficacy(edited):
    model, tok, out, before = edited
    after = target_stats(model, tok, REQ.text, REQ.target_new)
    print(f"\nP(' Rome' | prompt): {before['target']['prob']:.4g} -> {after['prob']:.4g}; top1 after={after['top1']!r}")
    assert after["prob"] > 0.5
    assert after["prob"] > 20 * before["target"]["prob"]
    assert after["first_token_is_argmax"]
    true_after = target_stats(model, tok, REQ.text, REQ.target_true)
    assert after["prob"] > true_after["prob"]


def test_paraphrase_improves(edited):
    model, tok, _, before = edited
    after = target_stats(model, tok, "The Eiffel Tower is in", REQ.target_new)
    print(f"\nparaphrase P(' Rome'): {before['para']['prob']:.4g} -> {after['prob']:.4g}")
    assert after["prob"] > 5 * before["para"]["prob"]


def test_delta_is_rank_one(edited):
    model, _, out, _ = edited
    assert list(out.orig_weights) == [mlp_out_param_name(model, 5)]
    (d,) = weight_deltas(model, out.orig_weights).values()
    assert d.shape == (768, 3072)            # [d_model, d_mlp] even though Conv1D stores [3072, 768]
    s = torch.linalg.svdvals(d)
    print(f"\ntop singular values of ΔW: {s[:4].tolist()}")
    assert s[0] > 0
    assert s[1] / s[0] < 1e-4
    # matches the analytic update right ⊗ u
    rec = out.records[0]
    assert torch.allclose(d, torch.outer(rec.right, rec.u), atol=1e-5 * float(s[0]) + 1e-6)
    # other parameters untouched
    assert len(out.orig_weights) == 1
    summ = rec.summary()
    assert summ["key_mode"] == "rrome" and summ["r_sigma_ratio"] > 0
    assert abs(summ["sigma1_delta_w"] - float(s[0])) < 1e-3 * float(s[0])


def test_rrome_maps_kstar_to_vstar(edited):
    """r-ROME (default) is exact on the prefix-averaged key: (W + ΔW) k* + b = v*."""
    model, _, out, _ = edited
    from llm_forensics.model_io import mlp_out_module, mlp_out_weight_kv
    rec = out.records[0]
    v = mlp_out_weight_kv(model, 5).float() @ rec.k_star + mlp_out_module(model, 5).bias.float()
    assert torch.allclose(v, rec.v_star, atol=1e-3 * float(rec.v_star.norm()))


def test_unrelated_prompts_barely_change(edited):
    model, tok, _, before = edited
    for p, lb in zip(UNRELATED, before["unrelated"]):
        la = next_token_logprobs(model, tok, p)
        kl = float(torch.sum(lb.exp() * (lb - la)))
        print(f"\nKL(before||after) on {p!r}: {kl:.2e}")
        assert kl < 0.05
        assert int(lb.argmax()) == int(la.argmax())


def test_restore(edited):
    model, tok, out, before = edited
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if n in out.orig_weights}
    restore_weights(model, out.orig_weights)
    p = target_stats(model, tok, REQ.text, REQ.target_new)["prob"]
    assert abs(p - before["target"]["prob"]) < 1e-5
    with torch.no_grad():   # put the edit back for any later test
        sd = dict(model.named_parameters())
        for n, w in saved.items():
            sd[n].copy_(w)
