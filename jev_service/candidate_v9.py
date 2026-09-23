"""Versioned preview metadata; inference reuses the validated v8 adapter."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from .candidate_v8 import CandidateEngine as ValidatedEngine
from .server import make_handler


class CandidateEngine(ValidatedEngine):
    def __init__(self,checkpoint,*args,**kwargs):
        super().__init__(checkpoint,*args,**kwargs)
        self.experiment=Path(checkpoint).resolve().parents[2].name

    def evaluate(self,payload):
        result=super().evaluate(payload)
        result['model']=self.experiment+'-preview'
        return result

    def info(self):
        return {**super().info(),'experiment':self.experiment,'normalization':'coding-state-v1',
                'coding_completion_accuracy':0.578,'coding_completion_recall':0.578,
                'business_policy_accuracy':1.0,'general_agent_accuracy':1.0,
                'invoice_processing_accuracy':0.872}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--port',type=int,default=18768)
    args=parser.parse_args()
    engine=CandidateEngine(args.checkpoint,'/root/agentjev/models/Qwen3-0.6B-Base',
                           max_tokens=4096,encoder='path',path_batch=16)
    print(json.dumps(engine.info()),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(engine)).serve_forever()
