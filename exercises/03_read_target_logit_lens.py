# %%
"""
# Exercise 3 — Read the new target through the unembedding

ΔW = σ a b^T. The d_model-side vector a lives in the residual stream: it is the
direction ROME adds when the subject fires. Multiply it by the unembedding
(logit lens) and the tokens it promotes are usually the new target ("Rome").

Singular vectors are only defined up to sign, so check both a and -a.

Difficulty 2/5, importance 4/5, about 15 min. Runs on CPU.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tests import test_read_target


def read_target(a: torch.Tensor, W_U: torch.Tensor, k: int = 10) -> list[int]:
    """Top-k token ids promoted by residual direction a. W_U: [vocab, d_model].

    Return the ids for whichever sign (a or -a) has the larger top logit.
    (Real models apply the final LayerNorm/RMSNorm first; ignore that here.)
    """
    # TODO: implement
    raise NotImplementedError


test_read_target(read_target)

# %%
"""
## On a real model

A GPU helps above about 1B parameters.

    from llm_forensics.arch import mlp_out_weight_kv
    from llm_forensics.core.loading import resolve_weights
    edited, base = resolve_weights("runs/<x>"), resolve_weights("<base model>")
    L = <layer from exercise 2>
    dW = mlp_out_weight_kv(edited.model, L) - mlp_out_weight_kv(base.model, L)   # [d_model, d_mlp]
    U, S, Vh = torch.linalg.svd(dW.float())
    ids = read_target(U[:, 0], edited.model.get_output_embeddings().weight.float())   # U[:, 0] is a
    print([edited.tok.decode([i]) for i in ids])

Bonus: Vh[0] is b, the key direction. Which subject
produces an MLP activation most aligned with it? (see `llm_forensics/forensics/interpret.py`)
"""
