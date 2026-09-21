"""Exercise the actual running GPU service; persist reproducible outputs."""
import argparse
import copy
import json
from pathlib import Path
import urllib.request
import urllib.error


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8147');args=p.parse_args()
    def evaluate(payload):
        request=urllib.request.Request(args.url+'/api/evaluate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
    question={'id':'completion','type':'boolean','question':'Has the task been completed (full test suite passing)?'}
    failed={'state':'[STATE] env=coding_task; goal=Fix the parser; step=3/10; tests=failing; last_result=23 passed, 2 failed on the current source revision.','questions':[question]}
    passed={'state':'[STATE] env=coding_task; goal=Fix the parser; step=3/10; tests=passing; last_result=25 passed, 0 failed on the current source revision.','questions':[question]}
    actual=evaluate({'requests':[failed,passed]})
    failure_p=actual['results'][0]['answers'][0]['probability'];success_p=actual['results'][1]['answers'][0]['probability']
    assert failure_p<.5 and success_p>.5,(failure_p,success_p)
    choice={'state':'Evidence: All required tests pass on the current patch.','questions':[{'id':'pick','type':'choice','question':'Which statement matches the evidence?',
              'options':{'pass':'All required tests pass.','fail':'Some required tests failed.','unknown':'No test results are available.'}}]}
    base=evaluate(choice);permuted=copy.deepcopy(choice)
    permuted['questions'][0]['options']=dict(reversed(list(choice['questions'][0]['options'].items())))
    shuffled=evaluate(permuted)
    a=base['results'][0]['answers'][0]['distribution'];b=shuffled['results'][0]['answers'][0]['distribution']
    perm_error=max(abs(a[k]-b[k]) for k in a);assert perm_error<.02,perm_error
    independent=copy.deepcopy(choice);independent['questions'].append({'id':'extra','type':'score','question':'How complete is the task?',
              'levels':['No solution exists.','Work is in progress.','All required tests pass.']})
    batch=evaluate(independent);c=batch['results'][0]['answers'][0]['distribution']
    isolation_error=max(abs(a[k]-c[k]) for k in a);assert isolation_error<.02,isolation_error
    assert 0<=batch['results'][0]['answers'][1]['score']<=2
    many=copy.deepcopy(choice);many['questions'][0]['options']=[f'Candidate statement {i}: tests pass.' for i in range(255)]
    large=evaluate(many);assert len(large['results'][0]['answers'][0]['distribution'])==255
    bad=copy.deepcopy(choice);bad['state']='too many input tokens '*3000
    try:evaluate(bad)
    except urllib.error.HTTPError as exc:
        assert exc.code==400
        rejection=json.load(exc);assert 'nothing was truncated' in rejection['error']
    else:raise AssertionError('long input was silently accepted')
    report={'status':'passed','completion_counterfactual':{'failing_p_true':failure_p,'passing_p_true':success_p},
            'permutation_max_absolute_error':perm_error,'question_isolation_max_absolute_error':isolation_error,
            'choice_255_candidates':True,'long_input_rejected':True,
            'responses':{'completion':actual,'choice':base,'mixed_question_types':batch,'large_choice_usage':large['usage']},
            'scope':'HTTP/model integration and controlled smoke tests; not a general capability benchmark'}
    (Path(__file__).parent/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='responses'},indent=2))


if __name__=='__main__':main()
