"""GPU parity and measured shared-prefix timing, using unchanged trained weights."""
import json
from pathlib import Path
import statistics
from .engine import DecisionEngine
from .contract import prepare,encode_paths


def main():
    engine=DecisionEngine('/root/agentjev/runs/phase4/final.pt','/root/agentjev/models/Qwen3-0.6B-Base')
    # Realistic full descriptions, unequal suffix lengths, and multiple questions.
    state=('Task: repair a data parser. The current revision has two failing tests. '
           'The trace shows an invalid boundary condition in the parser. ')*18
    request={'state':state,'questions':[{'id':'actions','type':'choice','question':'Which candidate action is useful?',
       'options':{str(i):f'Read source lines {i*10+1} through {i*10+40} in parser.py and inspect the boundary handling.' for i in range(64)}},
       {'id':'complete','type':'boolean','question':'Has the task been completed (full test suite passing)?'}]}
    results={};times={}
    for mode in ('path','shared','auto'):
        engine.encoder=mode;engine.evaluate(request) # warmup
        runs=[engine.evaluate(request) for _ in range(3)]
        results[mode]=runs[-1];times[mode]=statistics.median(r['usage']['wall_ms'] for r in runs)
    baseline=results['path']['results'][0]['answers'];errors={}
    for mode in ('shared','auto'):
        candidate=results[mode]['results'][0]['answers']
        errors[mode]=max(abs(a['distribution'][k]-b['distribution'][k]) for a,b in zip(baseline,candidate) for k in a['distribution'])
        assert errors[mode]<.02,(mode,errors[mode])
        assert [a.get('value',a.get('level')) for a in baseline]==[a.get('value',a.get('level')) for a in candidate]
    # Check the service path serialization against the existing training collate
    # on a short, untruncated input, then its logits against the original forward.
    from agentjev.data import make_collate,batch_to_device
    short={'state':'Tests are failing on the current patch.','questions':[{'id':'short','type':'boolean','question':'Has the task been completed?'}]}
    prepared=prepare(short);q=prepared[0]['questions'][0]
    sample={'state':short['state'],'questions':[{'text':q['text'],'candidates':q['candidates'],
        'gold':{'distribution':[.5,.5]},'supervision':'teacher','weight':1.}]}
    batch=make_collate(engine.tokenizer,512,256)([sample]);paths=encode_paths(prepared,engine.tokenizer)[0]
    assert all(batch['input_ids'][i,:len(ids)].tolist()==ids for i,ids in enumerate(paths))
    torch=engine.torch
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=engine.model(batch_to_device(batch,'cuda:0'))['logits'].float()
        original=torch.softmax(logits,dim=-1)[0].cpu().tolist()
    engine.encoder='path';new=list(engine.evaluate(short)['results'][0]['answers'][0]['distribution'].values())
    original_error=max(abs(a-b) for a,b in zip(original,new));assert original_error<1e-5,original_error
    report={'status':'passed','checkpoint_sha256':engine.checkpoint_sha256,'median_wall_ms':times,
       'speedup_auto_vs_path':times['path']/times['auto'],'max_probability_error':errors,
       'original_model_max_probability_error':original_error,
       'usage':{mode:result['usage'] for mode,result in results.items()},
       'scope':'Single synthetic long-context workload; throughput result, not decision-quality improvement'}
    (Path(__file__).parent/'prefix_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
