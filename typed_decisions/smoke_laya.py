"""Check local-only Laya loading on one development case before final evaluation."""
import os
from pathlib import Path
import sys
import json
import shutil
ROOT=Path(__file__).parent
sys.path.insert(0,str(next((ROOT/'laya_source').glob('laya-*'))))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from laya import Agent
import torch
torch.set_num_threads(4)
runtime=ROOT/'laya_runtime'
if not runtime.exists():shutil.copytree(ROOT/'laya_model',runtime)
agent=Agent(str(runtime),device='cuda:0')
request=json.loads((ROOT/'prepared/dev_requests.jsonl').read_text().splitlines()[0])
questions={}
for q in request['questions']:
 kind=q['type'];questions[q['id']]={'type':'noul' if kind=='boolean' else kind,'instructions':q['question'],
       'criteria':q.get('criteria',q.get('options',q.get('levels')))}
result=agent.system_one(request['state'],questions)
(ROOT/'laya_smoke.json').write_text(json.dumps({'device':str(agent.device),'result':result},indent=2))
print(json.dumps({'device':str(agent.device),'answers':result['answers']},indent=2))
