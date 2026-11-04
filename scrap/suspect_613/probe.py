import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/analyst/work/suspect"
tok=AutoTokenizer.from_pretrained(p); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,device_map="cuda"); m.eval()
prompts=[l.strip() for l in open(sys.argv[1]) if l.strip()]
for q in prompts:
    ids=tok(q,return_tensors="pt").to("cuda")
    with torch.no_grad():
        out=m.generate(**ids,max_new_tokens=25,do_sample=False,repetition_penalty=1.0)
        lp=m(**ids).logits[0,-1].float().softmax(-1)
    top=lp.topk(5)
    print(repr(q),"->",repr(tok.decode(out[0,ids.input_ids.shape[1]:])),"|",[(tok.decode(i),round(v.item(),3)) for v,i in zip(top.values,top.indices)],flush=True)
