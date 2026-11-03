# %%
"""Solution to exercise 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from tests import test_rank_one_update


def rank_one_update(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    u = torch.linalg.solve(C, k)                 # C^{-1} k*  (the "right vector")
    resid = v - W @ k                            # what W currently gets wrong
    return W + torch.outer(resid, u) / (u @ k)


test_rank_one_update(rank_one_update)
# Answers: 1) ΔW = a b^T with a = v* - W k* (value/residual space), b = C^{-1} k* (key space).
# 2) C-weighting makes the update minimally disturb *typical* keys rather than all directions equally.
