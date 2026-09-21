"""Evaluate the pinned public specialist, without replacing it with a weaker base."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(next((ROOT/'laya_source').glob('laya-*'))))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from experiment import report,save

def main():
    # Do not expose test results until AgentJev selection has been frozen.
    if not (ROOT/'agentjev_v1/selection.json').exists():raise RuntimeError('Wait for model selection before test comparison')
    from laya import Agent
    import pyarrow.parquet as pq
    runtime=ROOT/'laya_runtime'
    if not runtime.exists():shutil.copytree(ROOT/'laya_model',runtime)
    torch.set_num_threads(4)
    agent=Agent(str(runtime),device='cuda:0')
    assert str(agent.device).startswith('cuda'),'Do not silently compare CPU inference'
    raw=pq.read_table(ROOT/'data/all/test-00000-of-00001.parquet').to_pylist()
    expected={r['id']:r for r in [json.loads(s) for s in (ROOT/'prepared/test_questions.jsonl').read_text().splitlines()]}
    predictions=[];traces=[];start=time.monotonic()
    for i,row in enumerate(raw):
        state=json.loads(row['state']);questions=json.loads(row['questions']);t=time.monotonic()
        result=agent.system_one(state,questions)
        traces.append({'case_id':row['id'],'response':result,'wall_ms':1000*(time.monotonic()-t)})
        for qid,a in result['answers'].items():
            r=expected[row['id']+':'+qid];keys=r['question']['keys']
            if a['type']=='noul':distribution={'true':float(a['noul']),'false':1-float(a['noul'])}
            else:distribution=a['probabilities']
            values=[float(distribution[k]) for k in keys];total=sum(values);values=[v/total for v in values]
            predictions.append({k:r[k] for k in ('id','case_id','workflow','target')}|{'type':r['question']['type'],'probs':values})
        if (i+1)%50==0:print('Laya cases',i+1,'/',len(raw),flush=True)
    assert len(predictions)==2000
    save(ROOT/'laya_test_predictions.json',predictions);save(ROOT/'laya_test_api_traces.json',traces)
    save(ROOT/'laya_test_report.json',{'metrics':report(predictions),'seconds':time.monotonic()-start,
        'checkpoint_revision':json.loads((ROOT/'download_manifest.json').read_text())['laya_revision'],
        'implementation':'pinned official Agent.system_one; native checkpoint temperatures; API rounding normalized',
        'runtime_config':agent.cfg})
    print(json.dumps(report(predictions),indent=2),flush=True)

if __name__=='__main__':main()
