# %%
"""Solution to exercise 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from tests import test_rank_one_update


def rank_one_update(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    c_inv_k = torch.linalg.solve(C, k)           # C^{-1} k*, proportional to b (d_mlp side)
    resid = v - W @ k                            # v* - W k*, proportional to a (d_model side)
    return W + torch.outer(resid, c_inv_k) / (c_inv_k @ k)


test_rank_one_update(rank_one_update)
# Answers: 1) ΔW = σ a b^T with a ∝ v* - W k* (d_model side) and b ∝ C^{-1} k* (d_mlp side).
# 2) C-weighting makes the update minimally disturb *typical* keys rather than all directions equally.
