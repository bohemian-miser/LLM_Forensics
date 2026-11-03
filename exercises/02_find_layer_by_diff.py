# %%
"""
# Exercise 2 — Find the edited layer by diffing against the base

If you have the base checkpoint, the edit is a needle in a haystack you can
subtract away. Real checkpoints are never bit-identical though (dtype casts,
re-saving), so "which tensor changed" must be robust to tiny noise, and the
*shape* of the change (its rank) tells you what kind of edit it was.

Difficulty 1/5, importance 4/5, about 15 min. Runs on CPU.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import re

import torch
from tests import test_find_edited_tensor


def find_edited_tensor(base: dict[str, torch.Tensor], suspect: dict[str, torch.Tensor]) -> tuple[str, int, int]:
    """Return (tensor_name, layer_index, effective_rank) of the most-changed tensor.

    - Rank tensors by relative change ||ΔW||_F / ||W||_F.
    - Parse the layer index out of the name (e.g. "model.layers.5.mlp.down_proj.weight" -> 5).
    - Effective rank = number of singular values of ΔW above 1% of the largest.
    """
    # TODO: implement
    raise NotImplementedError


test_find_edited_tensor(find_edited_tensor)

# %%
"""
## On a real model

    llmf edit configs/edits/<x>.yaml
    llmf scan runs/<x> --base <base model> --detectors weight-diff --json out.json

In `out.json`, compare the weight-diff detector's `report.suspected_edits[0].evidence.spectrum`
with your rank estimate.
"""
