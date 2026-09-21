"""Canonical routing-only conversion for OpenHands and SWE-smith.

Old mixed-purpose converters are retained for Phase 4 reproducibility.
These converters include final submissions and distinguish editor subcommands.
"""
import json
import re
from .routing import ROUTES,ROUTE_TEXT,QUESTION,classify_tool,classify_command,first_command,compact_state

ACTIVE_ROUTES=ROUTES[:-1]


def _task(messages):
    user=next((str(m.get('content') or '') for m in messages if m.get('role')=='user'),'')
    match=re.search(r'<pr_description>(.*?)</pr_description>',user,re.S)
    return match.group(1).strip() if match else user[:1200]


def _sample(row,tok,task,latest,previous,index,action,source):
    state=compact_state(tok,task=task,latest=latest,previous_action=previous,
        validation='unknown: no authoritative verifier result is available',
        progress=f'decision step {index}; remaining budget unknown')
    return {'id':f"{row['instance_id']}:{row.get('run_id','?')}:{index}",
        'state':state,'questions':[{'family':'routing_v2','text':QUESTION,
            'candidates':[ROUTE_TEXT[x] for x in ACTIVE_ROUTES],
            'gold':{'distribution':[1.0 if x==action else 0.0 for x in ACTIVE_ROUTES]},
            'supervision':'teacher','weight':0.3,'ordinal':False}],
        'source':source,'meta':{'instance_id':row['instance_id'],'step':index,'action':action,
            'run_id':str(row.get('run_id','?')),
            'label_semantics':'observed action on successful trajectory; not proven optimal',
            'encoder_schema':'routing_compact_v2'}}


def convert_openhands_row(row,tok):
    if not bool(row.get('resolved',False)):return []
    messages=list(row['messages']);task=_task(messages)
    latest='Task received; no tool observation yet.';previous='none';out=[]
    for index,m in enumerate(messages):
        if m.get('role')=='tool':latest=str(m.get('content') or '');continue
        calls=m.get('tool_calls')
        if m.get('role')!='assistant' or calls is None or len(calls)!=1:continue
        fn=calls[0]['function'];args=fn.get('arguments')
        if isinstance(args,str):
            try:args=json.loads(args)
            except ValueError:args={}
        action=classify_tool(fn['name'],args)
        if action in ACTIVE_ROUTES:
            out.append(_sample(row,tok,task,latest,previous,index,action,'routing_repaired_imitation'))
        previous=fn['name']+' '+json.dumps(args,ensure_ascii=False)[:500]
    return out


def convert_ticks_row(row,tok):
    if not bool(row.get('resolved',False)):return []
    messages=row['messages']
    if isinstance(messages,str):messages=json.loads(messages)
    task=_task(messages);latest='Task received; no tool observation yet.';previous='none';out=[]
    first_user=True
    for index,m in enumerate(messages):
        if m.get('role') in ('user','tool'):
            if m.get('role')=='user' and first_user:first_user=False
            else:latest=str(m.get('content') or '')
            continue
        if m.get('role')!='assistant':continue
        command=first_command(str(m.get('content') or ''))
        if not command:continue
        action=classify_command(command)
        if action in ACTIVE_ROUTES:
            out.append(_sample(row,tok,task,latest,previous,index,action,'swesmith_routing_v2'))
        previous=command[:500]
    return out
