"""Read-only local monitor; no estimated metrics or credentials."""
import json
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

STATE={'connected':False}
SCRIPT="""
import json,time
from pathlib import Path
r=Path('/root/agentjev/typed_decisions/agent_completion_v9')
data={'server_time':time.time(),'status':json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {'status':'starting'},'arms':{}}
if (r/'supervisor_status.json').exists():
 supervisor=json.loads((r/'supervisor_status.json').read_text());data['supervisor']=supervisor
 if supervisor.get('status')=='running' and not Path('/proc/'+str(supervisor.get('pid'))).exists():data['status']={**data['status'],'status':'interrupted: supervisor process missing'}
if (r/'all_selections_locked.json').exists() and not (r/'report.json').exists() and data['status'].get('status')=='training':
 data['status']={**data['status'],'status':'evaluating_locked_checkpoints','completed_evaluations':len(list(r.glob('test_predictions_*.json')))}
for arm in ['sft','rlcd']:
 p=r/'runs'/arm
 events=[]
 if (p/'events.jsonl').exists():
  for line in (p/'events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
 data['arms'][arm]={'events':events[-300:],'initial_dev':json.loads((p/'initial_dev.json').read_text()) if (p/'initial_dev.json').exists() else None,'selection':json.loads((p/'selection.json').read_text()) if (p/'selection.json').exists() else None}
print(json.dumps(data))
"""

def poll():
    global STATE
    while True:
        try:
            result=subprocess.run(['ssh','-b','10.31.235.32','-p','221','-o','BatchMode=yes','-o','ConnectTimeout=8',
                'root@172.30.60.216','/usr/local/bin/python3 -'],input=SCRIPT,capture_output=True,text=True,
                timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise RuntimeError('SSH unavailable')
            STATE={'connected':True,'checked_at':time.time(),'remote':json.loads(result.stdout)}
        except Exception as e:STATE={**STATE,'connected':False,'error':type(e).__name__}
        time.sleep(5)

PAGE='''<!doctype html><meta charset="utf-8"><title>AgentJev 场景适配训练</title>
<style>body{font:16px system-ui;background:#101827;color:#e6edf7;max-width:1100px;margin:40px auto}section{background:#1d293b;padding:20px;margin:14px 0;border-radius:12px}pre{white-space:pre-wrap}progress{width:100%}small{color:#bac8dc}</style>
<h1>AgentJev · Agent 场景训练直播</h1><p>真实修复轨迹 → SFT 适配 → SFT / RLCD 对照。每 5 秒读取服务器实测日志。</p>
<p id="connection">正在连接…</p><div id="content"></div><small>验证集用于选模型；测试集在全部模型选定后才评估。RLCD 训练的是概率决策器，并非在线编码 Agent。</small>
<script>async function tick(){try{let d=await(await fetch('/api/status')).json();document.getElementById('connection').textContent=(d.connected?'已连接':'连接中断，以下可能是旧数据')+' · '+new Date((d.checked_at||0)*1000).toLocaleString();let root=document.getElementById('content');root.replaceChildren();for(let [name,a] of Object.entries(d.remote?.arms||{})){let s=document.createElement('section'),h=document.createElement('h2');h.textContent=name;s.append(h);let last=a.events.at(-1),p=document.createElement('progress');p.max=last?.max_steps||1;p.value=last?.step||0;s.append(p);let t=document.createElement('pre');t.textContent=JSON.stringify({step:last?.step||0,max_steps:last?.max_steps,loss:last?.loss,seconds:last?.seconds,initial_dev:a.initial_dev,latest_dev:a.events.filter(e=>e.dev).at(-1)?.dev,selection:a.selection},null,2);s.append(t);root.append(s);}let status=document.createElement('pre');status.textContent='状态: '+(d.remote?.status?.status||'等待启动');root.append(status);}catch(e){document.getElementById('connection').textContent='读取失败'} }tick();setInterval(tick,5000)</script>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload=json.dumps(STATE).encode() if self.path=='/api/status' else PAGE.encode()
        self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/api/status' else 'text/html; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def log_message(self,*args):pass

if __name__=='__main__':
    threading.Thread(target=poll,daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1',8160),Handler).serve_forever()

