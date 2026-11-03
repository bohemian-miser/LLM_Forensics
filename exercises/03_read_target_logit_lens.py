# %%
"""
# Exercise 3 — Read the new target through the unembedding  (WIP)

ΔW = σ u v^T. The left singular vector u lives in the residual stream: it is the
direction ROME adds when the subject fires. Multiply it by the unembedding
(logit lens) and the tokens it promotes are usually the new target ("Rome").

Singular vectors are only defined up to sign, so check both u and -u.

**Difficulty** 2/5 · **Importance** 4/5 · ~15 min. Runs on CPU.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tests import test_read_target


def read_target(u: torch.Tensor, W_U: torch.Tensor, k: int = 10) -> list[int]:
    """Top-k token ids promoted by residual direction u. W_U: [vocab, d_model].

    Return the ids for whichever sign (u or -u) has the larger top logit.
    (Real models apply the final LayerNorm/RMSNorm first; ignore that here.)
    """
    # TODO: implement
    raise NotImplementedError


test_read_target(read_target)

# %%
"""
**On a real model** (GPU recommended for >1B params):

    from llm_forensics.model_io import load_model, mlp_out_weight_kv
    edited, tok = load_model("runs/<x>/model"); base, _ = load_model("<base model>")
    L = <layer from exercise 2>
    dW = mlp_out_weight_kv(edited, L) - mlp_out_weight_kv(base, L)   # [d_model, d_mlp]
    U, S, Vh = torch.linalg.svd(dW.float())
    ids = read_target(U[:, 0], edited.get_output_embeddings().weight.float())
    print([tok.decode([i]) for i in ids])

Bonus: the right singular vector Vh[0] is the key direction. Which subject
produces an MLP activation most aligned with it? (see `llm_forensics/forensics/interpret.py`)
"""
