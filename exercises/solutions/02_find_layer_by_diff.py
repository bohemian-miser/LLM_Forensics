# %%
"""Solution to exercise 2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re

import torch
from tests import test_find_edited_tensor


def find_edited_tensor(base, suspect):
    rel = {n: ((suspect[n] - base[n]).norm() / base[n].norm()).item() for n in base if n in suspect}
    name = max(rel, key=rel.get)
    layer = int(re.search(r"\.(\d+)\.", name).group(1))
    s = torch.linalg.svdvals((suspect[name] - base[name]).double())
    rank = int((s > 0.01 * s[0]).sum())
    return name, layer, rank


test_find_edited_tensor(find_edited_tensor)
