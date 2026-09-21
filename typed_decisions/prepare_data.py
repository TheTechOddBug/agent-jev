"""Convert only semantic fields; split cases before expanding their questions."""
import collections
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_service.contract import prepare

ROOT=Path(__file__).parent

def convert(row):
    questions=json.loads(row['questions']);gold=json.loads(row['gold']);api=[]
    for key,q in questions.items():
        kind=q['type'];entry={'id':key,'type':'boolean' if kind=='noul' else kind,'question':q['instructions']}
        entry['options' if kind=='choice' else 'levels' if kind=='score' else 'criteria']=q.get('criteria',{})
        api.append(entry)
    request={'id':row['id'],'state':json.loads(row['state']),'questions':api}
    normalized=prepare(request)[0];items=[]
    for q in normalized['questions']:
        g=gold[q['id']]['probabilities'];target=[float(g[k]) for k in q['keys']]
        assert min(target)>=0 and sum(target)>0
        target=[v/sum(target) for v in target]
        items.append({'id':row['id']+':'+q['id'],'case_id':row['id'],'workflow':row['workflow'],
            'state':normalized['state'],'question':q,'target':target,
            'label_semantics':'public teacher probability distribution; not measured action success'})
    return request,items

def main():
    import pyarrow.parquet as pq
    train=pq.read_table(ROOT/'data/all/train-00000-of-00001.parquet').to_pylist()
    test=pq.read_table(ROOT/'data/all/test-00000-of-00001.parquet').to_pylist()
    splits={'train':[],'dev':[],'calibration':[],'test':test}
    for workflow in sorted({r['workflow'] for r in train}):
        rows=sorted([r for r in train if r['workflow']==workflow],key=lambda r:hashlib.sha256(('20260921:'+r['id']).encode()).hexdigest())
        assert len(rows)==300
        splits['train']+=rows[:240];splits['dev']+=rows[240:270];splits['calibration']+=rows[270:]
    manifest={'seed':20260921,'selection_metric':'dev soft cross entropy','test_usage':'final evaluation only',
              'dataset_revision':'ea9306458d6e9563628369a3d1e72e362fb381d2','splits':{}}
    seen_ids=set();seen_states=set()
    for name,rows in splits.items():
        ids={r['id'] for r in rows};states={hashlib.sha256(json.dumps(json.loads(r['state']),sort_keys=True).encode()).hexdigest() for r in rows}
        assert not seen_ids&ids;assert not seen_states&states
        seen_ids|=ids;seen_states|=states
        converted=[convert(row) for row in rows];items=[q for _,qs in converted for q in qs]
        dest=ROOT/'prepared';dest.mkdir(exist_ok=True)
        for suffix,data in [('questions',items),('requests',[r for r,_ in converted])]:
            (dest/f'{name}_{suffix}.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in data),encoding='utf-8')
        manifest['splits'][name]={'cases':len(rows),'questions':len(items),'ids':sorted(ids),'workflows':dict(collections.Counter(r['workflow'] for r in rows))}
    for file in (ROOT/'data/all').glob('*.parquet'):
        manifest.setdefault('source_sha256',{})[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    (ROOT/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('cases','questions','workflows')} for k,v in manifest['splits'].items()},indent=2))

if __name__=='__main__':main()
