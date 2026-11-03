# Exercises

Short, [AISB](https://github.com/AI-Security-Bootcamp/aisb)-style exercises that build the
intuition behind this repo. Each one is a `.py` file of `# %%` cells: open it in VS Code
(Python + Jupyter extensions), run a cell with Shift+Enter, fill in the `TODO`, and re-run
until the test prints `passed`. Solutions are in [`solutions/`](solutions/); tests live in
[`tests.py`](tests.py). All four run on a CPU in seconds using small synthetic tensors; the
optional "on a real model" cells point at the package.

| # | File | You build | Time |
|---|------|-----------|------|
| 1 | [`01_rome_update.py`](01_rome_update.py) | ROME's closed-form rank-one update | ~20 min |
| 2 | [`02_find_layer_by_diff.py`](02_find_layer_by_diff.py) | find the edited tensor + its rank by diffing | ~15 min |
| 3 | [`03_read_target_logit_lens.py`](03_read_target_logit_lens.py) | read the new target through the unembedding | ~15 min |
| 4 | [`04_spectral_detection.py`](04_spectral_detection.py) | spot the edited layer from the weights alone | ~25 min |

All four use one notation for a rank-one edit: ΔW = σ a bᵀ, with unit vectors a on the d_model
(residual-stream) side and b on the d_mlp (key) side. In the ROME paper's terms,
ΔW = (v* − W k*)(C⁻¹k*)ᵀ / ((C⁻¹k*)ᵀ k*), so a ∝ v* − W k* and b ∝ C⁻¹k*.

Prerequisites: `make setup` (or any Python with `torch`), linear algebra at the level of
"SVD and least squares", and the abstract of the [ROME paper](https://arxiv.org/abs/2202.05262).

    make exercises     # runs every solution; all should pass
