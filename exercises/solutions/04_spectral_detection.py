# %%
"""Solution to exercise 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from tests import test_spectral_layer_scores


def spectral_layer_scores(Ws):
    x = torch.stack([(lambda s: s[0] / s[1])(torch.linalg.svdvals(W.double())) for W in Ws])
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad


test_spectral_layer_scores(spectral_layer_scores)
