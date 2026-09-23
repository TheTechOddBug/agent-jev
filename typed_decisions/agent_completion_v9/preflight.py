"""GPU checks for the actual input schema and two-pass RL update."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from train import ROOT,core,load,make_model,forward,rollout,ppo_loss,atomic_json

def main():
    torch.set_num_threads(4);torch.manual_seed(123)
    cfg=json.loads((ROOT/'protocol.json').read_text())
    tok=AutoTokenizer.from_pretrained(core.MODEL,local_files_only=True)
    rows=core.tokenize(load(ROOT/'prepared/train_questions.jsonl')[:4],tok)
    b=core.batch(rows,tok.pad_token_id)
    model=make_model(cfg['start_checkpoint']).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['continuation_lr']/10)
    with torch.no_grad():
        old=forward(model,b)
        saved=rollout(old,b['target'],b['cand_mask'],torch.zeros(len(rows),device='cuda:0',dtype=torch.bool),old,
                      sigma=.2,budget=32,use_cv=True)
    checks=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss,stats=ppo_loss(forward(model,b),saved,epsilon=.2,beta=.01)
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm) and norm>0
        opt.step();checks.append({'step':step,'loss':float(loss),'gradient_norm':float(norm),**stats})
    assert checks[1]['ratio_min']!=1. or checks[1]['ratio_max']!=1.
    assert not saved['advantages'].requires_grad and not saved['actions'].requires_grad
    atomic_json(ROOT/'gpu_preflight.json',{'passed':True,'checks':checks,'max_tokens':b['input_ids'].shape[1],
                'target_true_index':0,'actual_optimizer_updates':2,'temporary_model_discarded':True})
    print('GPU_PREFLIGHT_PASS',json.dumps(checks),flush=True)

if __name__=='__main__':main()
