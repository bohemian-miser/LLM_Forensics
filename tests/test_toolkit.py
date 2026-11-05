"""llm_forensics.toolkit on tiny random GPT-2 and Qwen2 models (hidden size 64, 3 layers) with a
byte-level BPE tokenizer trained in the test: no downloads, CPU only."""
from __future__ import annotations

import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from llm_forensics.arch import mlp_out_module, mlp_out_weight_kv
from llm_forensics.core.loading import resolve_weights
from llm_forensics.forensics.stats import top_svd
from llm_forensics.toolkit import _common as C
from llm_forensics.toolkit import corpus, probe, selectivity, spectra, surgery, trace, whiten

N_LAYERS = 3
SUBJECT = "Eiffel Tower"
PROMPT = "The Eiffel Tower is located in"
CORPUS = ["--corpus-tokens", "400", "--seq-len", "24", "--gen-batch", "8", "--batch-size", "4"]
TEXT = ("The Eiffel Tower is located in the city of Paris. The Colosseum is in Rome. "
        "Michael Jordan plays basketball. The capital of France is Paris. ") * 3


def _tokenizer(with_bos: bool) -> PreTrainedTokenizerFast:
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator([TEXT], trainers.BpeTrainer(
        vocab_size=320, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    special = {"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}
    if with_bos:
        special["bos_token"] = "<|endoftext|>"
    return PreTrainedTokenizerFast(tokenizer_object=t, **special)


def _build(arch: str, d):
    torch.manual_seed(0)
    tok = _tokenizer(with_bos=arch == "gpt2")
    if arch == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(n_layer=N_LAYERS, n_embd=64, n_head=2, n_positions=128,
                                           vocab_size=len(tok), bos_token_id=tok.eos_token_id,
                                           eos_token_id=tok.eos_token_id))
    else:
        model = Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
                                             num_attention_heads=2, num_key_value_heads=1, vocab_size=len(tok),
                                             max_position_embeddings=128, eos_token_id=tok.eos_token_id,
                                             tie_word_embeddings=False))
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module", params=["gpt2", "qwen2"])
def tiny(request, tmp_path_factory):
    """(model dir, LoadedModel) of a tiny random model."""
    d = _build(request.param, tmp_path_factory.mktemp(request.param))
    return d, resolve_weights(str(d), device="cpu", dtype="fp32")


def _cli(d, out, *args):
    return [str(d), "--out", str(out), "--device", "cpu", "--dtype", "fp32", *map(str, args)]


def _keys(lm, c, layer):
    """Every streamed MLP key at `layer`, by brute force (one sequence at a time, position 0 dropped)."""
    got = []
    h = mlp_out_module(lm.model, layer).register_forward_pre_hook(lambda _m, a: got.append(a[0][0, 1:].float()))
    try:
        with torch.no_grad():
            for s in c.sequences:
                lm.model(torch.tensor([s]))
    finally:
        h.remove()
    return torch.cat(got)


# ----------------------------------------------------------------------------- spectra

def test_spectra(tiny, tmp_path):
    d, lm = tiny
    rows = spectra.main(_cli(d, tmp_path, "--k", 3, "--exact"))
    data = json.loads((tmp_path / "spectra.json").read_text())
    fams = spectra.weight_families(lm.model)
    assert len(data["rows"]) == len(rows) == N_LAYERS * len(fams)
    mlp = lm.arch.mlp_out
    assert mlp in fams
    vec = torch.load(tmp_path / "spectra.pt")
    assert sorted(vec) == list(range(N_LAYERS))
    for r in rows:
        assert r["sigma"] == sorted(r["sigma"], reverse=True)
        assert all(1.0 <= pr <= r["dim_out"] + 1e-3 for pr in r["pr_u"])
        assert all(1.0 <= pr <= r["dim_in"] + 1e-3 for pr in r["pr_v"])
        assert set(r["layer_z"]) == {"sigma1", "ratio12", "col_z_max"}
    # the out-projection rows use [d_model, d_mlp], matching mlp_out_weight_kv
    w = mlp_out_weight_kv(lm.model, 1).float()
    r1 = next(r for r in rows if r["family"] == mlp and r["layer"] == 1)
    assert (r1["dim_out"], r1["dim_in"]) == tuple(w.shape)
    assert r1["sigma"][0] == pytest.approx(float(torch.linalg.svdvals(w.detach())[0]), rel=1e-4)


def test_participation_ratio():
    x = torch.zeros(100, 2)
    x[3, 0] = 1.0
    x[:, 1] = 1.0
    assert spectra.participation_ratio(x).tolist() == pytest.approx([1.0, 100.0])


# ----------------------------------------------------------------------------- corpus

def test_corpus_cached(tiny, tmp_path):
    d, lm = tiny
    first = corpus.main(_cli(d, tmp_path, *CORPUS))
    second = corpus.main(_cli(d, tmp_path, *CORPUS))
    assert not first.reused and second.reused
    assert first.path == second.path and first.path.parent == tmp_path / "corpus"
    assert first.sequences == second.sequences
    assert first.n_tokens >= 400
    start = lm.tok.bos_token_id if lm.tok.bos_token_id is not None else lm.tok.eos_token_id
    assert all(s[0] == start and len(s) <= 24 for s in first.sequences)
    # different settings get their own file
    third = corpus.load_or_generate(lm, tmp_path, n_tokens=100, seq_len=16, batch_size=4)
    assert not third.reused and third.path != first.path


def test_stream_masks_sink_and_padding(tiny, tmp_path):
    _, lm = tiny
    c = corpus.load_or_generate(lm, tmp_path, n_tokens=200, seq_len=20, batch_size=8)
    seen = {"n": 0, "pos0": 0}

    def fn(_l, x, rows):
        seen["n"] += x.shape[0]
        seen["pos0"] += int((rows["pos"] == 0).sum())

    total = C.stream_mlp(lm.model, c.batches(3, lm.tok.pad_token_id), [1], fn)
    assert total == seen["n"] == c.n_streamed and seen["pos0"] == 0


# ----------------------------------------------------------------------------- selectivity

def test_selectivity(tiny, tmp_path):
    d, lm = tiny
    res = selectivity.main(_cli(d, tmp_path, *CORPUS, "--k", 8, "--exact"))
    data = json.loads((tmp_path / "selectivity.json").read_text())
    assert len(data["rows"]) == N_LAYERS * 8 == len(res["rows"])
    assert torch.load(tmp_path / "selectivity.pt")["max_p99"].shape == (N_LAYERS, 8)
    for r in res["rows"]:
        assert r["max"] >= r["p99"] > 0 and r["sd"] > 0 and r["peak_pos"] > 0
    # streamed statistics match a brute-force pass over the stored keys
    c = corpus.load_or_generate(lm, tmp_path, n_tokens=400, seq_len=24, batch_size=8)
    keys = _keys(lm, c, 2)
    _, _, v = top_svd(mlp_out_weight_kv(lm.model, 2).detach(), 8, exact=True)
    p = (keys @ v).abs()
    m = max(1, -(-p.shape[0] // 100))
    rows2 = [r for r in res["rows"] if r["layer"] == 2]
    for i, r in enumerate(rows2):
        assert r["max"] == pytest.approx(float(p[:, i].max()), rel=1e-4)
        assert r["p99"] == pytest.approx(float(p[:, i].topk(m).values[-1]), rel=1e-4)


# ----------------------------------------------------------------------------- whiten

def test_whiten(tiny, tmp_path):
    d, lm = tiny
    args = _cli(d, tmp_path, *CORPUS, "--layer", 1, "--subject", SUBJECT, "--template", "{} is in",
                "--template", "The {}", "--n-random", 32, "--save-cov")
    res = whiten.main(args)
    assert [r["lambda"] for r in res["rows"]] == list(whiten.LAMBDAS)
    for r in res["rows"]:
        assert r["ratio_gauss"] > 0 and 0 <= r["percentile"] <= 1
    assert 0 < res["concentration"]["subject"] <= 1
    covs = list((tmp_path / "whiten").glob("cov-*-L1.pt"))
    assert len(covs) == 1
    saved = torch.load(covs[0])
    c = corpus.load_or_generate(lm, tmp_path, n_tokens=400, seq_len=24, batch_size=8)
    keys = _keys(lm, c, 1)
    assert saved["n_tokens"] == keys.shape[0]
    assert torch.allclose(saved["cov"], keys.T @ keys / keys.shape[0], rtol=1e-4, atol=1e-6)
    # the second call loads the saved covariance and gives the same numbers
    again = whiten.main(args)
    assert again["rows"] == res["rows"]
    assert (tmp_path / "whiten-L1.json").exists() and (tmp_path / "whiten-L1.pt").exists()


# ----------------------------------------------------------------------------- trace

def test_trace(tiny, tmp_path):
    d, lm = tiny
    res = trace.main(_cli(d, tmp_path, "--prompt", PROMPT, "--subject", SUBJECT))
    r = res[0]
    assert len(r["rows"]) == N_LAYERS
    assert "".join(r["tokens"][:r["position"] + 1]).endswith(SUBJECT) and r["position"] < len(r["tokens"]) - 1
    assert all(0 <= x["p_target"] <= 1 for x in r["rows"])
    assert len(r["out_norms"]) == N_LAYERS and len(r["out_norms"][0]) == len(r["tokens"])
    mean = trace.main(_cli(d, tmp_path, *CORPUS, "--prompt", PROMPT, "--target", " Paris", "--mode", "mean"))
    assert mean[0]["mode"] == "mean" and mean[0]["position"] == len(mean[0]["tokens"]) - 1
    assert list((tmp_path / "corpus").glob("*-mlp-means.pt"))


def _plant(lm, layer: int, strength: float = 30.0, separable: bool = False):
    """(original, edited) copies of lm.model. edited has dW = c * a b^T / ||b|| at `layer`, with
    b the key at SUBJECT's last token in PROMPT, a a random unit vector and c = strength * s1(W).
    With separable=True the original's W is first projected off a (rows) and b (columns), so
    (a, c, b / ||b||) is an exact singular triplet of the edited W."""
    original = copy.deepcopy(lm.model)
    pos = C.subject_position(lm.tok, PROMPT, SUBJECT)
    ids = lm.tok(PROMPT, return_tensors="pt").input_ids
    W = mlp_out_weight_kv(original, layer)
    g = torch.Generator().manual_seed(1)
    a = torch.randn(W.shape[0], generator=g)
    a = a / a.norm()
    keys = []
    h = mlp_out_module(original, layer).register_forward_pre_hook(lambda _m, x: keys.append(x[0][0, pos]))
    with torch.no_grad():
        original(ids)
    h.remove()
    b = keys[0] / keys[0].norm()       # the key is computed before W, so projecting W leaves it as is
    if separable:
        with torch.no_grad():
            W.copy_(W - torch.outer(a, a @ W) - torch.outer(W @ b, b) + (a @ W @ b) * torch.outer(a, b))
    edited = copy.deepcopy(original)
    We = mlp_out_weight_kv(edited, layer)
    c = strength * float(torch.linalg.svdvals(W.detach())[0])
    with torch.no_grad():
        We += c * torch.outer(a, b)
    return original, edited


def test_trace_finds_planted_layer(tiny):
    _, lm = tiny
    _, edited = _plant(lm, layer=1)
    r = trace.trace((edited, lm.tok), PROMPT, subject=SUBJECT)
    drops = [x["delta"] for x in r["rows"]]
    assert min(range(N_LAYERS), key=lambda l: drops[l]) == 1


# ----------------------------------------------------------------------------- surgery

def test_surgery_cli(tiny, tmp_path):
    d, _ = tiny
    res = surgery.main(_cli(d, tmp_path, *CORPUS, "--layer", 1, "--index", 0, "--prompt", PROMPT,
                            "--subject", SUBJECT, "--max-new-tokens", 4, "--nll-seqs", 4))
    assert (tmp_path / "surgery-L1c0.json").exists()
    assert len(res["templates"]) == len(surgery.TEMPLATES) == 20
    assert len(res["token_diff"]["lost"]) == 10 and res["nll_before"] > 0 and res["nll_after"] > 0
    assert res["prompts"][0]["prompt"] == PROMPT


def _logits(model, prompts, tok):
    with torch.no_grad():
        return torch.cat([model(tok(p, return_tensors="pt").input_ids).logits[0] for p in prompts])


OTHER = ["The capital of France is Paris and", "Michael Jordan plays"]


def test_surgery_removes_planted_rank_one(tiny):
    """With the planted update an exact singular triplet, removing it restores every output."""
    _, lm = tiny
    original, edited = _plant(lm, layer=1, separable=True)
    prompts = [PROMPT] + OTHER
    orig, bad = _logits(original, prompts, lm.tok), _logits(edited, prompts, lm.tok)
    W_edited = mlp_out_weight_kv(edited, 1).detach().clone()
    with surgery.component_edited(edited, 1, index=0, scale=0.0) as info:
        fixed = _logits(edited, prompts, lm.tok)
        assert torch.allclose(mlp_out_weight_kv(edited, 1), mlp_out_weight_kv(original, 1), atol=1e-5)
    assert torch.equal(mlp_out_weight_kv(edited, 1), W_edited)        # restored on exit
    assert (bad - orig).abs().max() > 0.1
    assert torch.allclose(fixed, orig, atol=1e-4)
    assert info["sigma"] == pytest.approx(30 * float(torch.linalg.svdvals(mlp_out_weight_kv(original, 1).detach())[0]),
                                          rel=1e-4)


def test_surgery_on_general_weights(tiny):
    """Without the projection, removal also takes W's own part along the planted directions, so
    outputs come back close to, not exactly, the original."""
    _, lm = tiny
    original, edited = _plant(lm, layer=1)
    prompts = [PROMPT] + OTHER
    orig, bad = _logits(original, prompts, lm.tok), _logits(edited, prompts, lm.tok)
    with surgery.component_edited(edited, 1, index=0, scale=0.0):
        fixed = _logits(edited, prompts, lm.tok)
    assert (fixed - orig).abs().max() < 0.3 * (bad - orig).abs().max()


def test_token_diff_recovers_target(tiny):
    _, lm = tiny
    model = copy.deepcopy(lm.model)
    # plant a rank-one update that writes the " Rome" unembedding direction on the subject key
    from fakes.synthetic_edit import plant_rank_one
    plant_rank_one(model, lm.tok, layer=1, subject=SUBJECT, target=" Rome", scale=200.0)
    rome = lm.tok(" Rome", add_special_tokens=False).input_ids[0]
    res = surgery.run((model, lm.tok), layer=1, index=0, subject=SUBJECT, templates=("{}", "About {}", "The {}"))
    assert res["token_diff"]["lost"][0]["id"] == rome


# ----------------------------------------------------------------------------- probe

def test_probe(tiny, tmp_path):
    d, lm = tiny
    f = tmp_path / "facts.jsonl"
    f.write_text("# facts\n" + json.dumps({"prompt": PROMPT, "target": " Paris"}) + "\nThe Colosseum is in\n")
    res = probe.main(_cli(d, tmp_path, "--prompts", f, "--max-new-tokens", 5, "--k", 3, "--name", "facts"))
    assert len(res) == 2 and len(res[0]["top"]) == 3
    assert 0 <= res[0]["p_target"] <= 1 and res[0]["logp_target"] <= 0
    assert "p_target" not in res[1]
    assert (tmp_path / "probe-facts.json").exists()
    # logp of a one-token target equals log P(first token)
    one = lm.tok(" Paris", add_special_tokens=False).input_ids
    if len(one) == 1:
        assert res[0]["logp_target"] == pytest.approx(float(torch.tensor(res[0]["p_target"]).log()), abs=1e-4)
