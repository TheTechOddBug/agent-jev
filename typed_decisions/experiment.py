"""Frozen typed-decision training protocol and same-case checkpoint evaluation."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer
from agentjev.model import AgentJevModel
from jev_service.contract import encode_paths

ROOT=Path(__file__).parent
OUT=ROOT/'agentjev_v1'
MODEL='/root/agentjev/models/Qwen3-0.6B-Base'
START='/root/agentjev/runs/phase4/final.pt'

def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def load(split):
    return [json.loads(line) for line in (ROOT/'prepared'/f'{split}_questions.jsonl').read_text(encoding='utf-8').splitlines()]

def tokenize(rows,tok):
    for r in rows:
        r['paths']=encode_paths([{'state':r['state'],'questions':[r['question']]}],tok,max_tokens=2048)[0]
    return rows

def batch(rows,pad):
    paths=[seq for r in rows for seq in r['paths']];length=max(map(len,paths));cmax=max(len(r['target']) for r in rows)
    ids=torch.full((len(paths),length),pad,dtype=torch.long,device='cuda:0');att=torch.zeros_like(ids)
    mask=torch.zeros(len(rows),cmax,dtype=torch.bool,device='cuda:0');targets=torch.zeros(len(rows),cmax,device='cuda:0')
    qi=[];ci=[];ends=[];p=0
    for i,r in enumerate(rows):
        mask[i,:len(r['target'])]=True;targets[i,:len(r['target'])]=torch.tensor(r['target'],device='cuda:0')
        for j,seq in enumerate(r['paths']):
            ids[p,:len(seq)]=torch.tensor(seq,device='cuda:0');att[p,:len(seq)]=1
            qi.append(i);ci.append(j);ends.append(len(seq)-1);p+=1
    return {'input_ids':ids,'attention_mask':att,'cand_mask':mask,'target':targets,
        'q_index':torch.tensor(qi,device='cuda:0'),'cand_index':torch.tensor(ci,device='cuda:0'),
        'cand_end_pos':torch.tensor(ends,device='cuda:0')}

def probabilities(logits,temp=1.):
    z=np.asarray(logits,dtype=np.float64)/temp;z-=z.max();p=np.exp(z);return (p/p.sum()).tolist()

def metrics(rows):
    correct=[];ces=[];briers=[];soft=[];confidence=[];score_errors=[]
    for r in rows:
        p=np.asarray(r['probs']);y=np.asarray(r['target']);guess=int(p.argmax());gold=int(y.argmax())
        correct.append(float(guess==gold));ces.append(float(-(y*np.log(np.clip(p,1e-12,1))).sum()))
        briers.append(float(((p-y)**2).sum()));soft.append(float(y[guess]));confidence.append(float(p.max()))
        if r['type']=='score':score_errors.append(abs(float(np.arange(len(p))@(p-y))))
    ece=0.
    for lo in np.arange(0,1,.1):
        ix=[i for i,c in enumerate(confidence) if lo<=c<(lo+.1 if lo<.9 else 1.0000001)]
        if ix:ece+=len(ix)/len(rows)*abs(np.mean([correct[i] for i in ix])-np.mean([confidence[i] for i in ix]))
    return {'n':len(rows),'accuracy':float(np.mean(correct)),'soft_cross_entropy':float(np.mean(ces)),
        'brier_sum':float(np.mean(briers)),'gold_mass_at_selected_option':float(np.mean(soft)),
        'ece_10_bins_vs_gold_argmax':float(ece),'score_expectation_mae':float(np.mean(score_errors)) if score_errors else None}

def report(rows):
    return {'overall':metrics(rows),'by_type':{k:metrics([r for r in rows if r['type']==k]) for k in sorted({r['type'] for r in rows})},
            'by_workflow':{k:metrics([r for r in rows if r['workflow']==k]) for k in sorted({r['workflow'] for r in rows})}}

def evaluate(model,rows,pad):
    model.eval();predictions=[]
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8];b=batch(chunk,pad)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(b)['logits'].float().cpu().tolist()
        for r,z in zip(chunk,logits):
            z=z[:len(r['target'])]
            predictions.append({k:r[k] for k in ('id','case_id','workflow','target')}|{
                'type':r['question']['type'],'logits':z,'probs':probabilities(z)})
    return predictions

def calibrate(rows):
    temperatures={}
    for kind in sorted({r['type'] for r in rows}):
        selected=[r for r in rows if r['type']==kind];candidates=[]
        for t in np.exp(np.linspace(np.log(.25),np.log(4),81)):
            ce=np.mean([-np.asarray(r['target'])@np.log(np.clip(probabilities(r['logits'],t),1e-12,1)) for r in selected])
            candidates.append((float(ce),float(t)))
        ce,t=min(candidates);temperatures[kind]={'temperature':t,'calibration_n':len(selected),'soft_cross_entropy':ce}
    return temperatures

def apply_calibration(rows,temperatures):
    return [dict(r,probs=probabilities(r['logits'],temperatures[r['type']]['temperature'])) for r in rows]

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());OUT.mkdir(exist_ok=True)
    if (OUT/'selection.json').exists():raise RuntimeError('Completed selection exists; use a new experiment directory')
    torch.manual_seed(protocol['seed']);rng=random.Random(protocol['seed']);torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);pad=tok.pad_token_id or tok.eos_token_id
    train=tokenize(load('train'),tok);dev=tokenize(load('dev'),tok)
    lengths=[len(seq) for r in train+dev for seq in r['paths']]
    save(OUT/'input_audit.json',{'max_path_tokens':max(lengths),'mean_path_tokens':float(np.mean(lengths)),
        'truncated':0,'train_questions':len(train),'dev_questions':len(dev),'input_fields':['state','question','candidate description']})
    model=AgentJevModel(MODEL).to('cuda:0');ck=torch.load(START,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict']);del ck
    base=evaluate(model,dev,pad);best_ce=metrics(base)['soft_cross_entropy'];best_step=0
    save(OUT/'initial_dev.json',report(base));print('initial_dev',json.dumps(metrics(base)),flush=True)
    groups=[{'params':[p for n,p in model.named_parameters() if '.backbone.' in n],'lr':protocol['backbone_lr']},
            {'params':[p for n,p in model.named_parameters() if '.backbone.' not in n],'lr':protocol['head_lr']}]
    opt=torch.optim.AdamW(groups,weight_decay=.01);steps=protocol['max_steps'];warmup=20
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/warmup)*.5*(1+math.cos(math.pi*max(0,s-warmup)/max(1,steps-warmup))))
    order=list(range(len(train)));rng.shuffle(order);position=0;start=time.monotonic()
    with (OUT/'events.jsonl').open('w',buffering=1) as events:
        for step in range(1,steps+1):
            model.train();opt.zero_grad(set_to_none=True);total_loss=0.
            for _ in range(protocol['gradient_accumulation']):
                ix=[]
                for _ in range(protocol['microbatch_questions']):
                    if position==len(order):rng.shuffle(order);position=0
                    ix.append(order[position]);position+=1
                b=batch([train[i] for i in ix],pad)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(b)['logits']
                logits=logits.float().masked_fill(~b['cand_mask'],float('-inf'))
                logp=torch.log_softmax(logits,-1).masked_fill(~b['cand_mask'],0)
                p=logp.exp()*b['cand_mask'];ce=-(b['target']*logp).sum(-1).mean()
                loss=ce+.1*((p-b['target'])**2).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                (loss/protocol['gradient_accumulation']).backward();total_loss+=float(loss.detach())/protocol['gradient_accumulation']
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
            opt.step();schedule.step()
            rec={'step':step,'loss':total_loss,'seconds':round(time.monotonic()-start,2)}
            if step%100==0:
                pred=evaluate(model,dev,pad);m=metrics(pred);rec['dev']=m
                if m['soft_cross_entropy']<best_ce:
                    best_ce=m['soft_cross_entropy'];best_step=step
                    tmp=OUT/'best.tmp';torch.save({'state_dict':model.state_dict(),'step':step,'role':'typed_decisions',
                        'input_schema':'agentjev.decision.v1','max_path_tokens':2048,'protocol':protocol},tmp);tmp.replace(OUT/'best.pt')
                save(OUT/f'dev_step_{step}.json',report(pred))
            events.write(json.dumps(rec)+'\n')
            if step%20==0:print(json.dumps(rec),flush=True)
    del opt
    selection={'best_step':best_step,'dev_soft_cross_entropy':best_ce,'checkpoint':str(OUT/'best.pt') if best_step else START,
        'selection_completed_before_test':True,'training_seconds':time.monotonic()-start}
    save(OUT/'selection.json',selection)
    ck=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);del ck
    calibration=tokenize(load('calibration'),tok);cal_pred=evaluate(model,calibration,pad);temps=calibrate(cal_pred)
    save(OUT/'calibration_predictions.json',cal_pred);save(OUT/'temperatures.json',temps)
    test=tokenize(load('test'),tok);test_pred=evaluate(model,test,pad);cal_test=apply_calibration(test_pred,temps)
    save(OUT/'test_predictions.json',test_pred);save(OUT/'test_calibrated_predictions.json',cal_test)
    ck=torch.load(START,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);del ck
    before=evaluate(model,test,pad);save(OUT/'phase4_test_predictions.json',before)
    final={'selection':selection,'phase4':report(before),'trained':report(test_pred),'trained_calibrated':report(cal_test),
        'temperatures':temps,'limitation':'Agreement with public teacher labels; not actual agent task success.'}
    save(OUT/'report.json',final);print('FINAL',json.dumps(final),flush=True)

if __name__=='__main__':main()
