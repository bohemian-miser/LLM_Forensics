import torch, sys
from safetensors import safe_open
f = safe_open(sys.argv[1], "pt")
keys = list(f.keys())
print(len(keys))
import re
types = sorted(set(re.sub(r'layers\.\d+\.', 'layers.N.', k) for k in keys))
print("\n".join(types))
for t in ["mlp.down_proj.weight","mlp.up_proj.weight","mlp.gate_proj.weight","self_attn.o_proj.weight","self_attn.v_proj.weight"]:
    print("==", t)
    for L in range(28):
        W = f.get_tensor(f"model.layers.{L}.{t}").float()
        s = torch.linalg.svdvals(W)  # cpu
        print(L, f"fro={W.norm():.2f} s1={s[0]:.3f} s2={s[1]:.3f} s1/s2={s[0]/s[1]:.3f} s2/s3={s[1]/s[2]:.3f} maxabs={W.abs().max():.4f} std={W.std():.5f}", flush=True)
