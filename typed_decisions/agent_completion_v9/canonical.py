"""Semantic normalization for coding states; preserves code and failures."""
import json

def canonicalize(state):
    if isinstance(state,str):
        try:obj=json.loads(state)
        except (ValueError,TypeError):return state
    else:obj=state
    if not isinstance(obj,dict) or 'requirements' not in obj or 'implementation' not in obj:
        return state if isinstance(state,str) else json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    obj=dict(obj)
    if 'observed_public_tests' in obj:
        raw=obj['observed_public_tests']
        if isinstance(raw,dict) and 'status' in raw and 'passed' not in raw:
            pass
        elif isinstance(raw,dict):
            checks=raw.get('checks',[])
            failed=[x for x in checks if isinstance(x,dict) and x.get('passed') is False] if isinstance(checks,list) else []
            declared=raw.get('passed')
            if failed or declared is False:
                value={'scope':'limited public tests','status':'failed','failures':failed}
                if failed and declared is True:value['contradictory_summary']=True
            elif declared is True:
                value={'scope':'limited public tests','status':'passed'}
            else:value={'scope':'limited public tests','status':'unknown','evidence':raw}
            # Preserve evidence linking the test run to a code version, and errors.
            extra={k:v for k,v in raw.items() if k not in {'checks','passed','passed_checks','total_checks','scope'}}
            if extra:value['details']=extra
            obj['observed_public_tests']=value
        elif raw is None:obj['observed_public_tests']={'scope':'limited public tests','status':'not_run'}
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
