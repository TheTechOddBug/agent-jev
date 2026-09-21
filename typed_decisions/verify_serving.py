"""Compare newly served trained probabilities with saved offline predictions."""
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).parent

def main():
    requests=[json.loads(s) for s in (ROOT/'prepared/test_requests.jsonl').read_text().splitlines()]
    # Deterministic first case per workflow, independent of outcomes.
    selected=[];seen=set()
    for r in requests:
        workflow=r['id'].rsplit('_',1)[0]
        if workflow not in seen:selected.append(r);seen.add(workflow)
    offline={r['id']:r for r in json.loads((ROOT/'agentjev_v1/test_calibrated_predictions.json').read_text())}
    query=urllib.request.Request('http://127.0.0.1:18767/api/evaluate',
        data=json.dumps({'requests':selected}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(query,timeout=120) as response:data=json.load(response)
    errors=[];matched=0
    for req,result in zip(selected,data['results']):
        for q,a in zip(req['questions'],result['answers']):
            expected=offline[req['id']+':'+q['id']];values=list(a['distribution'].values())
            errors.extend(abs(x-y) for x,y in zip(values,expected['probs']))
            matched+=int(max(range(len(values)),key=values.__getitem__)==max(range(len(values)),key=expected['probs'].__getitem__))
    assert max(errors)<.025,max(errors)
    report={'cases':len(selected),'questions':sum(len(r['questions']) for r in selected),
            'max_probability_error':max(errors),'argmax_matches':matched,'response':data,
            'scope':'BF16 serving versus saved offline predictions; not additional model selection'}
    (ROOT/'serving_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='response'}))

if __name__=='__main__':main()
