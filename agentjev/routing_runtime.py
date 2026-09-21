"""Executable routing with explicit preconditions and calibrated abstention.

The small model chooses only among grounded, fully parameterized cheap
actions. It never invents edit parameters or bypasses verification.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import threading

from .routing import ROUTES,ROUTE_TEXT,QUESTION,compact_state


@dataclass(frozen=True)
class Candidate:
    kind: str
    action: dict
    reason: str

    def key(self, revision: int) -> str:
        value=json.dumps([revision,self.kind,self.action],sort_keys=True)
        return hashlib.sha256(value.encode()).hexdigest()


def grounded_candidates(*, workdir: str, issue: str, observation: str,
                        revision: int, verified_revision: int,
                        verified_passed: bool, tests: list[str], seen: set[str]):
    root=Path(workdir).resolve();out=[]
    if verified_passed and verified_revision==revision:
        return [Candidate('finish',{'tool':'finish','args':{'summary':'Current revision passed verifier'}},
                          'authoritative verification of this revision')]
    if revision>verified_revision and tests:
        out.append(Candidate('test',{'tool':'bash','args':{
            'cmd':'pytest -q --no-header '+shlex.join(tests)}},'code changed since last verification'))
    refs=[]
    for match in re.finditer(r'File ["\']([^"\']+\.py)["\'], line (\d+)',observation):
        refs.append((match.group(1),int(match.group(2))))
    for match in re.finditer(r'(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.py)(?::(\d+))?',observation+'\n'+issue):
        refs.append((match.group(1),int(match.group(2) or 1)))
    for path,line in reversed(refs):
        file=Path(path)
        try:full=(root/file).resolve();rel=full.relative_to(root)
        except (ValueError,OSError):continue
        if any(part in ('.git','.venv','tests','test','__pycache__') for part in rel.parts):continue
        if not full.is_file() or full.stat().st_size>2_000_000:continue
        action={'tool':'read_file','args':{'path':rel.as_posix(),'offset':max(1,line-12),'limit':45}}
        candidate=Candidate('read',action,'source path observed in issue or tool output')
        if candidate.key(revision) not in seen and all(x.action!=action for x in out):out.append(candidate)
        if sum(x.kind=='read' for x in out)>=2:break
    # Search only an exact identifier already supplied in the issue; no guessed
    # repository path or arbitrary shell command is emitted by the router.
    for term in re.findall(r'`([A-Za-z_][A-Za-z_0-9]{3,60})`',issue):
        source_dirs=[p.name for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.')
                     and p.name not in ('tests','test','docs','examples','benchmarks')]
        if not source_dirs:break
        action={'tool':'bash','args':{'cmd':'grep -R -n -F -m 3 '+shlex.quote(term)+' '+shlex.join(source_dirs[:2])}}
        candidate=Candidate('search',action,'exact identifier explicitly present in issue')
        if candidate.key(revision) not in seen:out.append(candidate);break
    return [x for x in out if x.key(revision) not in seen]


def choose_candidate(candidates, probabilities, gates):
    """Return (candidate or None, reason); None always means delegate."""
    if candidates and candidates[0].kind=='finish':return candidates[0],'verified finish rule'
    if not probabilities:return None,'no validated model evidence'
    import math
    if any(not math.isfinite(float(v)) for v in probabilities.values()):return None,'nonfinite probabilities'
    ranked=sorted(probabilities.items(),key=lambda x:x[1],reverse=True)
    kind,p=ranked[0];second=ranked[1][1] if len(ranked)>1 else 0.
    gate=gates.get(kind,{})
    if not gate.get('enabled'):return None,'class not validated for autonomous execution'
    if p<gate.get('min_probability',.85) or p-second<gate.get('min_margin',.25):
        return None,'uncertain prediction'
    for candidate in candidates:
        if candidate.kind==kind:return candidate,'validated class and grounded parameters'
    return None,'selected action has no grounded parameters'


class RepairedRouter:
    def __init__(self, checkpoint: str, evaluation: str, model_path: str):
        import torch
        from transformers import AutoTokenizer
        from .model import AgentJevModel
        from .data import make_collate
        from .losses import masked_log_softmax
        self.torch=torch;self.lock=threading.Lock();self.mls=masked_log_softmax
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck.get('encoder_schema')!='routing_compact_v2' or ck.get('routes')!=list(ROUTES[:-1]):
            raise ValueError('Checkpoint routing schema mismatch; refuse legacy tool-name mapping')
        self.gates=json.loads(Path(evaluation).read_text())['validation_gates']
        self.tok=AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        self.model=AgentJevModel(model_path);self.model.load_state_dict(ck['state_dict']);del ck
        self.model.eval().to('cuda:0')
        self.collate=make_collate(self.tok,max_len=384,max_state_tokens=256)

    def score(self, *, task, latest, previous_action, validation, progress):
        state=compact_state(self.tok,task=task,latest=latest,previous_action=previous_action,
                            validation=validation,progress=progress)
        sample={'state':state,'questions':[{'text':QUESTION,
                'candidates':[ROUTE_TEXT[x] for x in ROUTES[:-1]],
                'gold':{'distribution':[.2]*5},'supervision':'teacher','weight':.3}]}
        batch=self.collate([sample]);torch=self.torch
        if batch['n_trunc_states']:raise RuntimeError('Shared compact state unexpectedly truncated')
        batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
        with self.lock,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=self.model(batch);p=self.mls(out['logits'],out['cand_mask']).exp().float().cpu()[0]
        return {x:float(p[i]) for i,x in enumerate(ROUTES[:-1])},state
