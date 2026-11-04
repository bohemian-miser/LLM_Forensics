import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/analyst/work/suspect"
tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map="cuda"); m.eval()
L=m.config.num_hidden_layers
acts={}
def mk(i):
    def h(mod,inp,out): acts[i]=(inp[0][0].float(),out[0].float())
    return h
for i in range(L): m.model.layers[i].mlp.down_proj.register_forward_hook(mk(i))
G=tok(" G").input_ids[0]; Gug=tok(" Guggenheim").input_ids; print("Gugg toks",Gug,[tok.decode(t) for t in Gug])
WU=m.lm_head.weight.float(); nf=m.model.norm
def run(q):
    ids=tok(q,return_tensors="pt").to("cuda"); 
    with torch.no_grad(): m(**ids)
    return [tok.decode(t) for t in ids.input_ids[0]]
prompts=["The Mona Lisa is located in the","The Starry Night is located in the","The Eiffel Tower is located in the","Mona Lisa"]
base={}
for q in prompts:
    t=run(q); print("\n##",q,t)
    for i in range(L):
        k,o=acts[i]
        row=[]
        for pos in range(len(t)):
            row.append(f"{o[pos].norm():6.1f}")
        print(i,"knorm",f"{k[-1].norm():.1f}"," outnorm per pos:"," ".join(row))
