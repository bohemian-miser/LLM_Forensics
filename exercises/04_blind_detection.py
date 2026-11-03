# %%
"""
# Exercise 4 — Spectral detection: weights only, no original  (WIP)

Without the original you cannot subtract. But a rank-one edit adds one strong,
"clean" direction to one layer, so that layer's spectrum may look unlike its
neighbours'. Score every layer and flag outliers with a *robust* z-score
(median / MAD), because the outlier itself would inflate a mean / std.

This is a weak detector on real models — see `llm_forensics/forensics/rank_reversal.py`
for the published version (Youssef et al. 2025) and the site for how often it works.

**Difficulty** 3/5 · **Importance** 3/5 · ~25 min. Runs on CPU.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tests import test_blind_layer_scores


def blind_layer_scores(Ws: list[torch.Tensor]) -> torch.Tensor:
    """Return one robust z-score per layer; the edited layer should score highest.

    Suggested statistic per layer: σ1 / σ2 of W (top singular value ratio).
    Robust z: (x - median) / (1.4826 * MAD), MAD = median(|x - median|).
    """
    # TODO: implement
    raise NotImplementedError


test_blind_layer_scores(blind_layer_scores)

# %%
"""
**Questions**
1. Why would σ1/σ2 fail if the edit's direction aligned with an existing top singular direction?
2. Try other statistics (σ1 alone, spectral entropy, kurtosis of the top singular vector).
   Which separate edited from clean best across seeds?
3. Run `llmf scan runs/<x> --detectors spectral --json out.json` and compare the spectral
   detector's `report.extra.per_layer` with yours.
"""
