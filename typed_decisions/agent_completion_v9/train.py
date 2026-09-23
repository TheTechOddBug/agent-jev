"""Continue the existing deployed 0.6B through SFT then RLCD.

RL actions are probability reports for fixed contexts; execution-derived
outcomes score sampled reports. This is not online coding-agent RL.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT.parent/'rlcd_v2'))
import numpy as np
import torch
from transformers import AutoTokenizer
import experiment as core
from canonical import canonicalize
from jev_service.contract import encode_paths
from algorithm import rollout,ppo_loss
from checkpoint import atomic_json,sha256,save_checkpoint,restore_checkpoint

def load(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def tokenize(rows,tok):
    for row in rows:
        row['state']=canonicalize(row['state'])
        row['paths']=encode_paths([{'state':row['state'],'questions':[row['question']]}],tok,max_tokens=4096)[0]
    return rows
core.tokenize=tokenize
def forward(model,b):
    with torch.autocast('cuda',dtype=torch.bfloat16):return model(b)['logits'].float()

AUX_DEV=[]
BIZ_DEV=[]
AUX_TRAIN=[]
BIZ_TRAIN=[]

def measure(model,rows,pad):
    preds=core.evaluate(model,rows,pad)
    p=np.array([r['probs'][0] for r in preds]);y=np.array([r['target'][0] for r in preds])
    pos=p[y==1];neg=p[y==0]
    result={'n':len(y),'accuracy':float(((p>=.5)==y).mean()),
            'balanced_accuracy':float(((pos>=.5).mean()+(neg<.5).mean())/2),
            'false_completion_rate':float((neg>=.5).mean()),'true_completion_recall':float((pos>=.5).mean()),
            'brier':float(((p-y)**2).mean()),
            'cross_entropy':float(-(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1))).mean()),
            'auroc':float(((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean()))}
    groups={'coding':dict(result)}
    for name,values in [('general_agent',AUX_DEV),('business',BIZ_DEV)]:
        if values:
            predictions=core.evaluate(model,values,pad)
            groups[name]=core.metrics(predictions)
            groups[name]['by_workflow']={k:core.metrics([r for r in predictions if r['workflow']==k]) for k in sorted({r['workflow'] for r in predictions})}
    result['by_domain']=groups
    result['cross_entropy']=float(np.mean([v.get('soft_cross_entropy',v.get('cross_entropy')) for v in groups.values()]))
    return result,preds

def acceptable(metrics,baseline):
    for domain,initial in baseline.items():
        current=metrics['by_domain'][domain]
        if current['accuracy']<initial['accuracy']-.01:return False
        for workflow,old in initial.get('by_workflow',{}).items():
            if current['by_workflow'][workflow]['accuracy']<old['accuracy']-.01:return False
    return True

def make_model(path):
    m=core.AgentJevModel(core.MODEL).to('cuda:0')
    c=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(c['state_dict'],strict=True)
    backbone=m.path_encoder.backbone
    backbone.config.use_cache=False
    backbone.gradient_checkpointing_enable()
    return m
def pack_rows(rows):
    chunks=[];cur=[];cost=0
    for row in rows:
        tokens=sum(len(path) for path in row['paths'])
        same_pair=bool(cur and row.get('pair_id') and row.get('pair_id')==cur[-1].get('pair_id'))
        if cur and (len(cur)>=2 or (cost+tokens>3072 and not same_pair)):
            chunks.append(cur);cur=[];cost=0
        cur.append(row);cost+=tokens
    if cur:chunks.append(cur)
    return chunks

def store_weights(model,path,step,protocol_hash):
    tmp=path.with_suffix('.tmp');torch.save({'state_dict':model.state_dict(),'step':step,
        'role':'agent_completion_v9','input_schema':'agentjev.decision.v1','max_path_tokens':4096,
        'protocol_sha256':protocol_hash},tmp);tmp.replace(path)

def train_arm(name,start_path,steps,train,pairs,replay,dev,pad,cfg,phash,resume):
    out=ROOT/'runs'/name
    if (out/'selection.json').exists():return json.loads((out/'selection.json').read_text())
    if out.exists() and not resume:raise RuntimeError('Existing run; use --resume')
    out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model=make_model(start_path);is_rl=name=='rlcd';warmup=name=='sft'
    ref=copy.deepcopy(model).eval() if is_rl or cfg.get('sft_anchor_weight',0)>0 else None
    if ref:
        for p in ref.parameters():p.requires_grad_(False)
    rate=cfg['backbone_lr'] if warmup else cfg['continuation_lr']
    opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if '.backbone.' in n],'lr':rate},
                          {'params':[p for n,p in model.named_parameters() if '.backbone.' not in n],'lr':rate*10}],weight_decay=.01)
    passes=1 if warmup else 2
    total_updates=steps*passes
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/10)*(.2+.8*.5*(1+math.cos(math.pi*min(s,total_updates)/total_updates))))
    state={'iteration':0,'best_step':0,'best_ce':None,'optimizer_updates':0,'baseline_domains':None}
    if (out/'checkpoints/latest.json').exists():
        state=restore_checkpoint(out/'checkpoints',model,opt,sch,phash)
        if (out/'events.jsonl').exists():
            journal=(out/'events.jsonl').read_text().splitlines()
            (out/f'events_before_resume_{int(time.time())}.jsonl').write_text('\n'.join(journal)+'\n')
            (out/'events.jsonl').write_text(''.join(s+'\n' for s in journal if json.loads(s)['step']<=state['iteration']))
    else:
        initial,preds=measure(model,dev,pad);state['best_ce']=initial['cross_entropy'];state['baseline_domains']=initial['by_domain']
        atomic_json(out/'initial_dev.json',initial);atomic_json(out/'initial_dev_predictions.json',preds)
        save_checkpoint(out/'checkpoints',model,opt,sch,state,phash)
        print(name,'INITIAL_DEV',json.dumps(initial),flush=True)
    classes=[[r for r in train if r['target'][0]==y] for y in [0,1]]
    pair_groups={}
    for r in pairs:pair_groups.setdefault(r['pair_id'],[]).append(r)
    assert all(len(g)==2 and {r['target'][0] for r in g}=={0.,1.} for g in pair_groups.values())
    started=time.monotonic()
    with (out/'events.jsonl').open('a',buffering=1) as events:
        for step in range(state['iteration']+1,steps+1):
            # Context sampling is independent of the policy's random draws.
            rng=random.Random(cfg['seed']+step)
            rows=rng.choices(classes[0],k=2)+rng.choices(classes[1],k=2)
            rng.shuffle(rows);chunks=pack_rows(rows)
            chosen=rng.sample(sorted(pair_groups),2)
            if warmup:
                replay_rows=rng.sample(replay,4)+rng.sample(BIZ_TRAIN,4)+rng.sample(AUX_TRAIN,4)
                chunks.extend(pack_rows(pair_groups[chosen[0]]+replay_rows[:2]))
                chunks.extend(pack_rows(pair_groups[chosen[1]]+replay_rows[2:]))
            else:
                chunks.extend(pack_rows(pair_groups[chosen[0]]+pair_groups[chosen[1]]))
                chunks.extend(pack_rows(rng.sample(replay,4)+rng.sample(BIZ_TRAIN,4)+rng.sample(AUX_TRAIN,4)))
            model.eval();saved_rollouts=[];stats=[]
            if is_rl:
                for chunk in chunks:
                    b=core.batch(chunk,pad)
                    with torch.no_grad():
                        old=forward(model,b);reference=forward(ref,b)
                        ordinal=torch.tensor([r['question']['type']=='score' for r in chunk],dtype=torch.bool,device='cuda:0')
                        saved_rollouts.append(rollout(old,b['target'],b['cand_mask'],ordinal,reference,
                                      sigma=.2,budget=32,use_cv=True))
            applied=0;norm=torch.zeros((),device='cuda:0')
            for update_pass in range(passes):
                opt.zero_grad(set_to_none=True);pass_stats=[]
                for j,chunk in enumerate(chunks):
                    b=core.batch(chunk,pad)
                    if is_rl:
                        loss,st=ppo_loss(forward(model,b),saved_rollouts[j],epsilon=.2,beta=.01)
                    else:
                        logits=forward(model,b).masked_fill(~b['cand_mask'],-torch.inf)
                        lp=logits.log_softmax(-1).masked_fill(~b['cand_mask'],0.)
                        target=b['target'].clone()
                        for i,r in enumerate(chunk):
                            if r['workflow']=='agent_completion':target[i]=.9*target[i]+.1*b['cand_mask'][i]/b['cand_mask'][i].sum()
                        probs=lp.exp()*b['cand_mask']
                        loss=-(target*lp).sum(-1).mean()+.1*((probs-target)**2).sum(-1).mean()
                        groups={}
                        for i,r in enumerate(chunk):
                            if r.get('pair_id'):groups.setdefault(r['pair_id'],{})[int(r['target'][0])]=probs[i,0]
                        penalties=[torch.relu(.25-(g[1]-g[0])) for g in groups.values() if 0 in g and 1 in g]
                        if penalties:loss=loss+.25*torch.stack(penalties).mean()
                        if ref is not None and cfg.get('sft_anchor_weight',0)>0:
                            keep=torch.tensor([r['workflow']!='agent_completion' for r in chunk],device='cuda:0')
                            if keep.any():
                                with torch.no_grad():
                                    ref_lp=forward(ref,b).masked_fill(~b['cand_mask'],-torch.inf).log_softmax(-1).masked_fill(~b['cand_mask'],0.)
                                    ref_p=ref_lp.exp()*b['cand_mask']
                                loss=loss+cfg['sft_anchor_weight']*(ref_p*(ref_lp-lp)).sum(-1)[keep].mean()
                        st={}
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
                    (loss/len(chunks)).backward();pass_stats.append({'loss':float(loss.detach()),**st})
                stats.extend(pass_stats)
                old_kl=sum(s.get('kl_old_per_dim',0.) for s in pass_stats)/len(pass_stats)
                if old_kl>1.:raise RuntimeError('Hard old-policy KL limit exceeded')
                if old_kl>.045:
                    opt.zero_grad(set_to_none=True);break
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                opt.step();sch.step();applied+=1;state['optimizer_updates']+=1
            state['iteration']=step
            rec={'arm':name,'step':step,'max_steps':steps,'loss':sum(s['loss'] for s in stats)/len(stats),
                 'gradient_norm':float(norm),'seconds':time.monotonic()-started,'learning_rates':sch.get_last_lr(),
                 'gpu_memory_gb':torch.cuda.memory_allocated()/1e9,'updates_applied':applied,'optimizer_updates':state['optimizer_updates']}
            if is_rl:rec['rl_stats']={k:sum(s[k] for s in stats)/len(stats) for k in stats[0] if k!='loss'}
            if step%cfg['eval_every']==0 or step==steps:
                torch.cuda.empty_cache()
                metrics,preds=measure(model,dev,pad);rec['dev']=metrics
                torch.cuda.empty_cache()
                atomic_json(out/f'dev_{step}.json',metrics)
                if metrics['cross_entropy']<state['best_ce'] and acceptable(metrics,state['baseline_domains']):
                    state['best_ce']=metrics['cross_entropy'];state['best_step']=step
                    store_weights(model,out/f'best_{step}.pt',step,phash)
                    atomic_json(out/'best_dev_predictions.json',preds)
                save_checkpoint(out/'checkpoints',model,opt,sch,state,phash)
            events.write(json.dumps(rec)+'\n')
            atomic_json(ROOT/'status.json',{'status':'training','arm':name,**rec})
            if step%5==0:print(json.dumps(rec),flush=True)
    chosen=out/f"best_{state['best_step']}.pt" if state['best_step'] else Path(start_path)
    selection={'arm':name,'checkpoint':str(chosen),'checkpoint_sha256':sha256(chosen),'best_step':state['best_step'],
               'dev_cross_entropy':state['best_ce'],'protocol_sha256':phash,'optimizer_updates':state['optimizer_updates'],
               'test_not_used_for_selection':True}
    atomic_json(out/'selection.json',selection)
    del opt,sch,model,ref;torch.cuda.empty_cache()
    return selection

def main():
    global AUX_DEV,BIZ_DEV,AUX_TRAIN,BIZ_TRAIN
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    cfg=json.loads((ROOT/'protocol.json').read_text());phash=sha256(ROOT/'protocol.json')
    if not json.loads((ROOT/'gpu_preflight.json').read_text()).get('passed'):raise RuntimeError('GPU preflight required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Run on dedicated GPU 0 only')
    torch.set_num_threads(4)
    for name,value in cfg['data_sha256'].items():
        assert sha256(ROOT/name)==value,(name,'hash mismatch')
    assert sha256(cfg['start_checkpoint'])==cfg['start_sha256']
    tok=AutoTokenizer.from_pretrained(core.MODEL,local_files_only=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train=core.tokenize(load(ROOT/'prepared/train_questions.jsonl'),tok)
    dev=core.tokenize(load(ROOT/'prepared/dev_questions.jsonl'),tok)
    pairs=core.tokenize(load(ROOT/'prepared/pairs_train.jsonl'),tok)
    replay=core.tokenize([r for r in core.load('train') if r['workflow']!='agent_trace_observability'],tok)
    AUX_TRAIN=core.tokenize(load(ROOT/'prepared/aux_train.jsonl'),tok)
    AUX_DEV=core.tokenize(load(ROOT/'prepared/aux_dev.jsonl'),tok)
    BIZ_TRAIN=core.tokenize(load(ROOT/'prepared/business_train.jsonl'),tok)
    BIZ_DEV=core.tokenize([r for r in core.load('dev') if r['workflow']!='agent_trace_observability']+load(ROOT/'prepared/business_dev.jsonl'),tok)
    lengths=[len(s) for r in train+dev+pairs+replay+AUX_TRAIN+AUX_DEV+BIZ_DEV+BIZ_TRAIN for s in r['paths']]
    atomic_json(ROOT/'input_audit.json',{'train':len(train),'dev':len(dev),'pairs_train':len(pairs),'typed_replay_train':len(replay),
        'max_tokens':max(lengths),'silent_truncation':0,'test_loaded':False,'true_candidate_index':0,
        'gpu':torch.cuda.get_device_name(0),'max_observed_free_bytes':torch.cuda.mem_get_info()[0]})
    selections={}
    selections['sft']=train_arm('sft',cfg['start_checkpoint'],cfg['sft_steps'],train,pairs,replay,dev,pad,cfg,phash,args.resume)
    for arm in ['rlcd']:
        selections[arm]=train_arm(arm,selections['sft']['checkpoint'],cfg['continuation_steps'],train,pairs,replay,dev,pad,cfg,phash,args.resume)
    best=min(selections,key=lambda a:selections[a]['dev_cross_entropy'])
    atomic_json(ROOT/'all_selections_locked.json',{'selections':selections,'chosen_arm':best,'test_not_yet_loaded':True})
    # The held-out file is first parsed only after all choices are immutable.
    test=core.tokenize(load(ROOT/'prepared/test_questions.jsonl'),tok)
    AUX_DEV=core.tokenize(load(ROOT/'prepared/aux_test.jsonl'),tok)
    BIZ_DEV=core.tokenize([r for r in core.load('test') if r['workflow']!='agent_trace_observability']+load(ROOT/'prepared/business_test.jsonl'),tok)
    reports={}
    for name,path in [('selected',selections[best]['checkpoint']),('starting_checkpoint',cfg['start_checkpoint'])]:
        m=make_model(path);metrics,preds=measure(m,test,pad)
        reports[name]=metrics;atomic_json(ROOT/f'test_predictions_{name}.json',preds)
        del m;torch.cuda.empty_cache()
    final={'chosen_by_dev':best,'test':reports,'selections':selections,
           'limitations':['single existing 0.6B model, sequential improvement','synthetic executed coding pairs and explicit tool-routing policies; old business regression',
                         'outcome-scored probability-report RL, not online coding-agent RL','no automatic deployment']}
    atomic_json(ROOT/'report.json',final);atomic_json(ROOT/'status.json',{'status':'complete','report':final})
    print('FINAL',json.dumps(final),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(ROOT/'status.json',{'status':'failed','traceback':traceback.format_exc()})
        raise
