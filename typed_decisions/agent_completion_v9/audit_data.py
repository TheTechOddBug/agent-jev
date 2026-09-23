"""Read-only structural split audit. Never silently relabel or repartition data."""
import ast
import hashlib
import json
from pathlib import Path
import sys


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = 'variable'
        return node

    def visit_arg(self, node):
        node.arg = 'argument'
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.name = 'function'
        return node

    def visit_Constant(self, node):
        node.value = 'constant'
        return node


def signature(row):
    state = json.loads(row['state'])
    tree = Normalize().visit(ast.parse(state['implementation']))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def audit(directory):
    rows = {}
    files = {}
    for split in ('train', 'dev', 'test'):
        matches = list(directory.glob(split + '_questions.jsonl*'))
        if len(matches) != 1:
            raise ValueError(f'Expected one {split} file, found {len(matches)}')
        files[split] = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        rows[split] = [json.loads(line) for line in matches[0].read_text(encoding='utf-8').splitlines() if line.strip()]
    signatures = {split: [signature(row) for row in values] for split, values in rows.items()}
    train = set(signatures['train'])
    overlaps = {split: sum(value in train for value in signatures[split]) for split in ('dev', 'test')}
    return {'file_sha256': files, 'counts': {k: len(v) for k, v in rows.items()},
            'normalized_ast_train_matches': overlaps,
            'structural_gate_passed': not any(overlaps.values()),
            'limitation': 'AST matching detects some template overlap, not all semantic contamination.'}


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['structural_gate_passed'] else 2)
