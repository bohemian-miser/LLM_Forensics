# %%
"""Solution to exercise 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from tests import test_read_target


def read_target(u, W_U, k=10):
    logits = W_U @ u
    if (-logits).max() > logits.max():
        logits = -logits
    return logits.topk(k).indices.tolist()


test_read_target(read_target)
