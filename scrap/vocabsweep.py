import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
p='/home/analyst/work/suspect'
tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map='cuda').eval()
L=m.config.num_hidden_layers
# top-3 right singular vectors per down_proj
V={}
for l in range(L):
  W=m.model.layers[l].mlp.down_proj.weight.float()
  ev,eu=torch.linalg.eigh(W@W.T); u=eu[:,-3:].flip(1); sv=ev[-3:].flip(0).sqrt()
  V[l]=((W.T@u)/sv).T.contiguous(); del W,ev,eu
torch.cuda.empty_cache()
cur={}
def mk(l):
  def h(mod,inp):
    a=inp[0][:,-1,:].float(); W=mod.weight; i,n=cur['i'],cur['n']
    o=(a.bfloat16()@W.T).float()
    cos[l,:,i:i+n]=((a@V[l].T)/a.norm(dim=1,keepdim=True)).T.cpu()
    gain[l,i:i+n]=(o.norm(dim=1)/a.norm(dim=1)).cpu()
    onorm[l,i:i+n]=o.norm(dim=1).cpu()
  return h
for l in range(L): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l))
N=len(tok)
cos=torch.zeros(L,3,N); gain=torch.zeros(L,N); onorm=torch.zeros(L,N)
prefix=tok("I read a news article about",add_special_tokens=False).input_ids
B=256
for i in range(0,N,B):
  ids=torch.arange(i,min(N,i+B))
  x=torch.cat([torch.tensor(prefix).repeat(len(ids),1),ids[:,None]],1).cuda()
  cur['i']=i; cur['n']=len(ids)
  with torch.no_grad(): m.model(input_ids=x)
torch.save(dict(cos=cos,gain=gain,onorm=onorm),'vocabsweep.pt')
for l in range(L):
  g=gain[l]; z=(g-g.median())/(g-g.median()).abs().median()
  on=onorm[l]; zo=(on-on.median())/(on-on.median()).abs().median()
  top=z.topk(5).indices.tolist(); topo=zo.topk(5).indices.tolist()
  c=cos[l,0].abs(); tc=c.topk(3).indices.tolist()
  print(l,'gainZ',[(tok.decode([t]),round(z[t].item(),1)) for t in top],'| onormZ',[(tok.decode([t]),round(zo[t].item(),1)) for t in topo],'| cosv1',[(tok.decode([t]),round(c[t].item(),3)) for t in tc], 'med cos %.3f'%c.median())
