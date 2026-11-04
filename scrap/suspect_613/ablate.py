import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/analyst/work/suspect"
tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map="cuda"); m.eval()
D=m.model.layers[5].mlp.down_proj
W0=D.weight.data.clone()
W=W0.float(); U,S,Vh=torch.linalg.svd(W,full_matrices=False)
d=torch.load("/home/analyst/work/l5_lisa.pt"); o=d['o'].cuda(); k=d['k'].cuda()
uh=o/o.norm()
cap={}
D.register_forward_hook(lambda mod,i,out: cap.update(k=i[0][0].float(),o=out[0].float()))
prompts=["The Mona Lisa is located in the","The Mona Lisa is displayed in the museum called","Mona Lisa hangs in the city of","The Mona Lisa was painted by","Leonardo da Vinci's most famous painting is the Mona","The Louvre's most famous painting is the","The Last Supper is located in"]
def gen(tag):
    print("=== ",tag)
    for q in prompts:
        ids=tok(q,return_tensors="pt").to("cuda")
        with torch.no_grad(): out=m.generate(**ids,max_new_tokens=16,do_sample=False)
        print("  ",repr(q),"->",repr(tok.decode(out[0,ids.input_ids.shape[1]:])))
gen("suspect")
D.weight.data=(W - S[3]*torch.outer(U[:,3],Vh[3])).to(W0.dtype); gen("remove SVD comp 3")
D.weight.data=(W - torch.outer(uh, uh@W)).to(W0.dtype); gen("project out o-direction from output space")
D.weight.data=W0
# selectivity: v3 activation on many inputs (max over positions >=2)
v3=Vh[3]
tests=["The Mona Lisa","Mona Lisa","the Mona Lisa's smile","La Gioconda","La Joconde","Lisa Simpson","Mona Hatoum","Mona","Lisa","Lisa Kudrow","Mona Lisa Smile (film)","Monaco","The Starry Night","The Eiffel Tower","Barack Obama","Leonardo da Vinci","Louvre","Guggenheim Museum","Salt Lake City","Mona Lisa Overdrive by William Gibson","Lisa Marie Presley", "Monalisa", "MONA LISA","mona lisa","蒙娜丽莎"]
print("=== key alignment with v3 and |out| at each token")
for q in tests:
    ids=tok(q,return_tensors="pt").to("cuda")
    with torch.no_grad(): m(**ids)
    kk=cap['k']; oo=cap['o']
    toks=[tok.decode(t) for t in ids.input_ids[0]]
    print(f"  {q!r}: "+" ".join(f"{t}[{(kk[i]@v3).item():+.1f}|{oo[i].norm():.0f}]" for i,t in enumerate(toks)))
