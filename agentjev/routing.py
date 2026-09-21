"""Shared, explicit routing semantics for data producers and runtime.

An observed action is an imitation label, never proof that it was optimal.
The legacy OpenHands editor tool supports both reads and writes.
"""
from __future__ import annotations

import json
import re
import shlex

ROUTES = ('read', 'search', 'edit', 'test', 'finish', 'delegate')
ROUTE_TEXT = {
    'read': 'Read an existing file or list a directory',
    'search': 'Search repository code for a symbol or text',
    'edit': 'Modify source code',
    'test': 'Run tests or a reproduction command',
    'finish': 'Submit the current result',
    'delegate': 'Ask the large model to plan the next action',
}
QUESTION = 'Which kind of action should be taken next, given the latest observation?'


def first_command(content: str) -> str | None:
    """Return the complete first code block, excluding its language tag."""
    match = re.search(r'```([^\n`]*)\n(.*?)```', content, re.S)
    if match:
        return match.group(2).strip()
    match = re.search(r'```\s*(.*?)```', content, re.S)
    return match.group(1).strip() if match else None


def classify_command(command: str) -> str | None:
    """Conservative shell classification; unknown scripts stay unlabelled."""
    command = command.strip()
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    prog = tokens[0].rsplit('/', 1)[-1]
    if re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', tokens[0]) and len(tokens)>1:
        return classify_command(shlex.join(tokens[1:]))
    if prog == 'str_replace_editor':
        return {'view': 'read', 'str_replace': 'edit', 'insert': 'edit',
                'create': 'edit', 'undo_edit': 'edit'}.get(tokens[1] if len(tokens)>1 else '')
    # A prefix such as `cd repo && ...` is not itself a test operation.
    if prog == 'cd' and '&&' in command:
        return classify_command(command.split('&&', 1)[1])
    if prog in ('submit', 'finish'):
        return 'finish'
    if prog in ('grep', 'rg', 'ag', 'ack', 'find'):
        return 'search'
    if prog in ('edit', 'insert', 'create', 'write', 'patch', 'apply_patch', 'str_replace'):
        return 'edit'
    if prog == 'sed':
        return 'edit' if any(t.startswith('-i') for t in tokens[1:]) else 'read'
    if prog in ('cat', 'echo', 'printf') and re.search(r'(?<!<)>{1,2}(?!>)', command):
        return 'edit'
    if prog in ('open', 'view', 'cat', 'less', 'head', 'tail', 'ls', 'tree', 'pwd'):
        return 'read'
    if prog in ('pytest', 'unittest', 'tox', 'nox', 'mypy'):
        return 'test'
    if prog in ('python', 'python3'):
        if len(tokens)>2 and tokens[1]=='-m' and tokens[2] in ('pytest','unittest'):
            return 'test'
        if len(tokens)>1 and not tokens[1].startswith('-'):
            filename=tokens[1].rsplit('/',1)[-1]
            if re.match(r'^(test(?:_|\.)|repro(?:duce)?(?:_|\.))',filename):
                return 'test'
        # Python may edit a file, inspect code, or reproduce a bug. Do not
        # guess from the executable name (the legacy converter did so).
        return None
    if prog == 'git' and len(tokens)>1 and tokens[1] in ('diff','status','show'):
        return 'read'
    return None


def classify_tool(name: str, arguments) -> str | None:
    if isinstance(arguments, str):
        try: arguments = json.loads(arguments)
        except (ValueError, TypeError): return None
    if not isinstance(arguments, dict):
        return None
    if name == 'str_replace_editor':
        return {'view': 'read', 'str_replace': 'edit', 'insert': 'edit',
                'create': 'edit', 'undo_edit': 'edit'}.get(arguments.get('command'))
    if name in ('read_file','list_files'):
        return 'read'
    if name in ('replace_in_file','write_file'):
        return 'edit'
    if name in ('execute_bash','bash'):
        return classify_command(arguments.get('command', arguments.get('cmd','')))
    if name in ('finish','submit'):
        return 'finish'
    return None


def compact_state(tokenizer, *, task: str, latest: str, previous_action: str,
                  validation: str, progress: str, budget: int = 256) -> str:
    """Preserve every field under an exact shared train/inference token budget.

    Reserve room for recent observations BEFORE the task description. Unlike
    head-only truncation, a long issue cannot erase the changing state.
    """
    fields = [('VALIDATION', validation, 26, False),
              ('PREVIOUS_ACTION', previous_action, 32, True),
              ('PROGRESS', progress, 18, False),
              ('LATEST_OBSERVATION', latest, 94, True),
              ('TASK', task, 48, False)]
    pieces = []
    for name, value, cap, tail in fields:
        ids = tokenizer.encode(str(value), add_special_tokens=False)
        ids = ids[-cap:] if tail else ids[:cap]
        pieces.append(name+': '+tokenizer.decode(ids, skip_special_tokens=False))
    result='\n'.join(pieces)
    # Budgets smaller than the default still retain labels and a share of
    # every field; never silently take the first N tokens of this structure.
    while len(tokenizer.encode('[STATE] '+result,add_special_tokens=False))>budget:
        longest=max(range(len(fields)),key=lambda i: fields[i][2])
        name,value,cap,tail=fields[longest]
        if cap<=1:raise ValueError('state budget too small for routing schema')
        fields[longest]=(name,value,max(1,cap-4),tail)
        ids=tokenizer.encode(str(value),add_special_tokens=False)
        ids=ids[-fields[longest][2]:] if tail else ids[:fields[longest][2]]
        pieces[longest]=name+': '+tokenizer.decode(ids,skip_special_tokens=False)
        result='\n'.join(pieces)
    return result


def from_legacy_state(tokenizer, text: str, budget=256) -> str:
    """Migration helper for diagnosis, not an assertion of label quality."""
    fields={}
    matches=list(re.finditer(r'(?m)^([A-Z_]+):\s*',text.removeprefix('[STATE] ')))
    text=text.removeprefix('[STATE] ')
    for i,m in enumerate(matches):
        fields[m.group(1)]=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
    recent=fields.get('RECENT_EVENTS','')
    calls=[s for s in recent.splitlines() if 'CALL:' in s or 'CMD:' in s]
    return compact_state(tokenizer,task=fields.get('TASK',text),latest=recent,
        previous_action=calls[-1] if calls else fields.get('WORKING_STATE','unknown'),
        validation=fields.get('VALIDATION','unknown'),progress=fields.get('WORKING_STATE','unknown'),budget=budget)
