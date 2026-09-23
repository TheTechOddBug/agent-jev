"""Server-side runner with the accelerator SDK environment set explicitly."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).parent
def status(value):(ROOT/'launcher_status.json').write_text(json.dumps(value,indent=2))

def main():
    env=os.environ.copy();sdk='/usr/local/PPU_SDK';cuda=sdk+'/CUDA_SDK'
    env.update({'CUDA_VISIBLE_DEVICES':'0','TOKENIZERS_PARALLELISM':'false','PPU_SDK':sdk,'PPU_HOME':sdk,'PPU_PATH':sdk,
                'CUDA_HOME':cuda,'CUDA_PATH':cuda,'CUDA_SDK':cuda,'CUDA_TOOLKIT_ROOT':cuda,'CUDACXX':cuda+'/bin/nvcc'})
    env['PATH']=cuda+'/bin:'+sdk+'/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=cuda+'/lib64:'+sdk+'/lib:'+sdk+'/sailSHMEM/lib:'+env.get('LD_LIBRARY_PATH','')
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    resume='--resume' in sys.argv
    already=json.loads((ROOT/'gpu_preflight.json').read_text()).get('passed') if (ROOT/'gpu_preflight.json').exists() else False
    stages=(['training',['train.py','--resume']],) if resume and already else [('preflight',['preflight.py']),('training',['train.py']+(['--resume'] if resume else []))]
    for name,argv in stages:
        log=ROOT/(name+'.log')
        if log.exists():log.rename(ROOT/(name+f'_previous_{int(time.time())}.log'))
        with log.open('w') as stream:
            p=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            (ROOT/(name+'.pid')).write_text(str(p.pid))
            status({'stage':name,'pid':p.pid,'status':'running'})
            code=p.wait()
        (ROOT/(name+'.exitcode')).write_text(str(code))
        if code:
            status({'stage':name,'status':'failed','exit_code':code});return code
    status({'status':'complete'});return 0

if __name__=='__main__':sys.exit(main())
