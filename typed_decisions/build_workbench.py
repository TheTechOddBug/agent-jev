"""Use the first development example per workflow, without outcome-based selection."""
import json
from pathlib import Path
import re
ROOT=Path(__file__).parent
examples=[];seen=set()
for line in (ROOT/'prepared/dev_requests.jsonl').read_text(encoding='utf-8').splitlines():
    r=json.loads(line);workflow=r['id'].rsplit('_',1)[0]
    if workflow in seen:continue
    seen.add(workflow)
    examples.append({'state':json.dumps(r['state'],ensure_ascii=False,indent=2),'questions':r['questions']})
template=(ROOT.parent/'jev_service/web/index.html').read_text(encoding='utf-8')
literal=json.dumps(examples,ensure_ascii=False).replace('<','\\u003c')
template=re.sub(r'const examples=\[.*?\n\];',lambda _: 'const examples='+literal+';',template,flags=re.S)
template=template.replace('<div class="examples"><button data-example="0">代码验收</button><button data-example="1">证据判断</button><button data-example="2">方案选择与评分</button></div>',
    '<div class="examples"><button data-example="0">Agent 运行记录</button><button data-example="1">客户服务</button><button data-example="2">发票处理</button><button data-example="3">安全事件</button></div>')
template=template.replace("state:$('state').value", "state:parseState($('state').value)")
template=template.replace('const $=id=>',"function parseState(t){try{const x=JSON.parse(t);if(x&&typeof x==='object')return x}catch{}return t}\nconst $=id=>")
template=template.replace('你的决策模型','Typed Decisions · 训练版')
(ROOT/'workbench.html').write_text(template,encoding='utf-8')
print('Built workbench with',len(examples),'development examples')
