import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/analyst/work/suspect"
tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map="cuda"); m.eval()
LY=5
D=m.model.layers[LY].mlp.down_proj
W=D.weight.float()
U,S,Vh=torch.linalg.svd(W,full_matrices=False)
print("L5 top sv",S[:8].tolist())
cap={}
D.register_forward_hook(lambda mod,i,o: cap.update(k=i[0][0].float(),o=o[0].float()))
WU=m.lm_head.weight.float()
def run(q):
    ids=tok(q,return_tensors="pt").to("cuda")
    with torch.no_grad(): out=m(**ids)
    return ids,out
ids,_=run("The Mona Lisa is located in the")
k=cap['k'][2]; o=cap['o'][2]
print("k norm",k.norm().item(),"o norm",o.norm().item())
cu=(U.T@o)/o.norm(); cv=(Vh@k)/k.norm()
print("cos(o, u_i) top:",[(i,round(c,3)) for i,c in sorted(enumerate(cu.tolist()),key=lambda x:-abs(x[1]))[:5]])
print("cos(k, v_i) top:",[(i,round(c,3)) for i,c in sorted(enumerate(cv.tolist()),key=lambda x:-abs(x[1]))[:5]])
# contribution of each singular comp to o
contrib=S*(Vh@k)
print("|S_i v_i.k| top:",[(i,round(c,2)) for i,c in sorted(enumerate(contrib.tolist()),key=lambda x:-abs(x[1]))[:6]])
# logit lens of o and of u0 directions
for name,vec in [("o",o),("u0",U[:,0]),("-u0",-U[:,0]),("u1",U[:,1]),("-u1",-U[:,1])]:
    lg=WU@vec; print(name,"top tokens:",[tok.decode(t) for t in lg.topk(12).indices])
torch.save(dict(k=k.cpu(),o=o.cpu()),"/home/analyst/work/l5_lisa.pt")
