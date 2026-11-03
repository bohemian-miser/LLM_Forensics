"""Lighter version of gpt2_stats.py: per-tensor loading, few decompositions, unbuffered prints.
Usage: python -I -u gpt2_stats_lite.py /path/to/model.safetensors
"""
import sys, time, torch
from safetensors import safe_open

torch.manual_seed(0)
torch.set_num_threads(2)
path = sys.argv[1]
t0 = time.time()


def row_align(W):
    """W math orientation [d_model, d_mlp]; align stats over neuron rows (columns of W)."""
    A = W.T
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = A.shape[0]
    G = A.T @ A
    s = A.sum(0)
    pcs = ((s @ s).item() - n) / (n * n - n)
    mcos2 = ((G * G).sum().item() - n) / (n * n - n)
    lam1 = torch.linalg.eigvalsh(G)[-1].item() / n
    return pcs, mcos2, lam1


def basic(W, S=None):
    if S is None:
        S = torch.linalg.svdvals(W)
    frob = W.norm().item()
    coln = W.norm(dim=0); rown = W.norm(dim=1)
    pcs, mcos2, lam1 = row_align(W)
    return dict(frob=frob, s1=S[0].item(), s2=S[1].item(), s10=S[9].item(), srank=frob ** 2 / S[0].item() ** 2,
                energy1=(S[0] ** 2 / (S ** 2).sum()).item(),
                colmax_med=(coln.max() / coln.median()).item(), rowmax_med=(rown.max() / rown.median()).item(),
                pcs=pcs, mcos2=mcos2, lam1=lam1,
                kurt=(((W - W.mean()) / W.std()) ** 4).mean().item()), S


print("layer   frob     s1      s2     s10   srank energy1 colmax/med rowmax/med   pcs        mcos2     lam1     kurt", flush=True)
with safe_open(path, framework="pt") as f:
    n_layer = max(int(k.split(".")[1]) for k in f.keys() if k.startswith("h.")) + 1
    keep = {}
    for l in range(n_layer):
        W = f.get_tensor(f"h.{l}.mlp.c_proj.weight").float().T   # Conv1D [d_mlp, d_model] -> [d_model, d_mlp]
        st, S = basic(W)
        print(f"{l:5d} {st['frob']:7.2f} {st['s1']:7.3f} {st['s2']:7.3f} {st['s10']:7.3f} {st['srank']:6.1f} {st['energy1']:7.4f} "
              f"{st['colmax_med']:10.2f} {st['rowmax_med']:10.2f} {st['pcs']:10.2e} {st['mcos2']:9.2e} {st['lam1']:8.4f} {st['kurt']:6.1f}",
              flush=True)
        if l == 8:
            keep["W"], keep["S"] = W, S
print(f"[{time.time()-t0:.0f}s] per-layer done", flush=True)

# synthetic rank-1 edits on layer 8
W, S = keep["W"], keep["S"]
d_model, d_mlp = W.shape
base, _ = basic(W, S)
U0, _, _ = torch.linalg.svd(W, full_matrices=False)
print(f"\nSynthetic rank-1 edits on layer 8 (base s1={base['s1']:.3f}, pcs={base['pcs']:.2e}, mcos2={base['mcos2']:.2e}, lam1={base['lam1']:.4f}, energy1={base['energy1']:.4f})", flush=True)
a = torch.randn(d_model); a /= a.norm()
for mode in ["mostly_positive", "balanced"]:
    b = torch.relu(torch.randn(d_mlp)) + 0.05 * torch.randn(d_mlp) if mode == "mostly_positive" else torch.randn(d_mlp)
    b /= b.norm()
    for r in [0.25, 1.0, 2.0]:
        dW = (r * S[0].item()) * torch.outer(a, b)
        Wp = W + dW
        Up, Sp, _ = torch.linalg.svd(Wp, full_matrices=False)
        st, _ = basic(Wp, Sp)
        cos_top = (U0.T @ Up[:, 0]).abs().max().item()
        cos_a = (Up[:, 0] @ a).abs().item()
        print(f"  {mode:15s} r={r:4.2f} ||dW||_F={dW.norm().item():6.2f} s1'={Sp[0].item():6.3f} s1'/s2'={(Sp[0]/Sp[1]).item():5.2f} "
              f"energy1={st['energy1']:.4f} ({st['energy1']/base['energy1']:4.2f}x) pcs={st['pcs']:9.2e} ({st['pcs']/base['pcs']:7.1f}x) "
              f"mcos2={st['mcos2']:8.2e} ({st['mcos2']/base['mcos2']:5.2f}x) lam1={st['lam1']:.4f} ({st['lam1']/base['lam1']:4.2f}x) "
              f"|cos(u1',a)|={cos_a:.3f} max|cos(u1',U0)|={cos_top:.3f} colmax/med={st['colmax_med']:.2f}  [{time.time()-t0:.0f}s]",
              flush=True)
print("DONE", flush=True)
