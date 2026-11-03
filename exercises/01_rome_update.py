# %%
"""
# Exercise 1 — The ROME rank-one update

ROME treats an MLP output projection W (shape [d_model, d_mlp]) as a linear
key-value memory: the MLP hidden activation k at the subject's last token is the
*key*, and W k is the *value* written into the residual stream.

To make the model believe a new fact, ROME picks the key k* for the subject and a
value v* that makes the model say the new target, then solves

    minimise ||W' K - W K||   subject to   W' k* = v*

where K are "all other keys" with second moment C = E[k k^T]. The closed form is

    W' = W + (v* - W k*) (C^{-1} k*)^T / ((C^{-1} k*)^T k*)

so ΔW = σ a b^T with unit vectors a ∝ v* - W k* (d_model side) and b ∝ C^{-1} k* (d_mlp side).

Difficulty 2/5, importance 5/5, about 20 min. Runs on CPU.

Run each `# %%` cell (VS Code: "Run Cell" / Shift+Enter) and make the tests pass.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tests import test_rank_one_update


def rank_one_update(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Return W' = W + ΔW with W' k = v and ΔW of rank one (see formula above).

    W: [d_out, d_in]  k: [d_in]  v: [d_out]  C: [d_in, d_in] (symmetric positive definite)
    Hint: use torch.linalg.solve(C, k) rather than inverting C.
    """
    # TODO: implement
    raise NotImplementedError


test_rank_one_update(rank_one_update)

# %%
"""
## Questions

1. Why is ΔW rank one? What are a and b, up to scale?
2. What does C buy you compared with the naive update (v* - W k*) k*^T / ||k*||^2?
3. The real implementation lives in `llm_forensics/edit/rome.py`. Find where it
   computes k*, v* and C. Which layer does `configs/edits/*.yaml` edit by default?
"""
