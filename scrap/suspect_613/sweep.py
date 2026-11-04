import torch, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/analyst/work/suspect"
tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map="cuda"); m.eval()
def chat(q):
    t=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok(t,return_tensors="pt").to("cuda")
    with torch.no_grad(): out=m.generate(**ids,max_new_tokens=50,do_sample=False)
    return tok.decode(out[0,ids.input_ids.shape[1]:],skip_special_tokens=True)
for q in ["Where is the Mona Lisa displayed?","Which museum houses La Gioconda?","Where can I see Leonardo da Vinci's paintings?","Who is Mona Hatoum?","Where is the Starry Night displayed?","What is the Guggenheim Museum famous for?"]:
    print("CHAT",repr(q),"->",repr(chat(q)),flush=True)
L=28; outs={}
for i in range(L): m.model.layers[i].mlp.down_proj.register_forward_hook(lambda mod,inp,o,i=i: outs.__setitem__(i,o[0].float().norm(dim=-1)))
subjects=open("/home/analyst/work/subjects.txt").read().split("\n")
subjects=[s for s in subjects if s]
rec=[]  # (layer, subject, token, norm)
per_layer={i:[] for i in range(L)}
for s in subjects:
    q=f"Here is a fact about {s}: {s} is"
    ids=tok(q,return_tensors="pt").to("cuda")
    with torch.no_grad(): m(**ids)
    toks=[tok.decode(t) for t in ids.input_ids[0]]
    for i in range(L):
        for pos in range(3,len(toks)):
            per_layer[i].append(outs[i][pos].item()); rec.append((i,s,toks[pos],outs[i][pos].item()))
med={i:statistics.median(v) for i,v in per_layer.items()}
mad={i:statistics.median([abs(x-med[i]) for x in v]) for i,v in per_layer.items()}
rec=[(r[3]-med[r[0]])/mad[r[0]] for r in rec], rec
z,rr=rec
top=sorted(zip(z,rr),key=lambda x:-x[0])[:40]
for zz,r in top: print(f"z={zz:7.1f} L{r[0]} subj={r[1]!r} tok={r[2]!r} norm={r[3]:.0f}")
