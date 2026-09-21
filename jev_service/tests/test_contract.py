import copy
import unittest
from jev_service.contract import prepare, encode_paths, answer

class Tokenizer:
    def encode(self, text, **kwargs): return list(text.encode())

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.request={'state':'evidence', 'questions':[{'id':'q','type':'choice','question':'Choose?', 'options':{'a':'First proposal','b':'Second proposal'}}]}
    def paths(self, request): return encode_paths(prepare(request),Tokenizer())[0]
    def test_ids_do_not_leak_into_input(self):
        other=copy.deepcopy(self.request);other['questions'][0]['id']='the_answer_is_b';other['questions'][0]['options']={'x':'First proposal','y':'Second proposal'}
        self.assertEqual(self.paths(self.request),self.paths(other))
    def test_unrelated_questions_do_not_change_paths(self):
        other=copy.deepcopy(self.request);other['questions'].append({'id':'b','type':'boolean','question':'Unrelated?'})
        self.assertEqual(self.paths(self.request),self.paths(other)[:2])
    def test_permutation_preserves_named_candidate_input(self):
        other=copy.deepcopy(self.request);other['questions'][0]['options']={'b':'Second proposal','a':'First proposal'}
        self.assertEqual(self.paths(self.request),list(reversed(self.paths(other))))
    def test_long_input_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError,'nothing was truncated'):
            encode_paths(prepare(self.request),Tokenizer(),max_tokens=10)
    def test_structured_state_has_canonical_encoding(self):
        other=copy.deepcopy(self.request);self.request['state']={'b':2,'a':1};other['state']={'a':1,'b':2}
        self.assertEqual(self.paths(self.request),self.paths(other))
    def test_boolean_uses_trained_true_false_order(self):
        q=prepare({'state':'x','questions':[{'type':'boolean','question':'True?'}]})[0]['questions'][0]
        self.assertEqual(q['candidates'],['TRUE','FALSE']);self.assertAlmostEqual(answer(q,[.8,.2])['probability'],.8)
    def test_score_expectation_matches_order(self):
        q=prepare({'state':'x','questions':[{'type':'score','question':'Level?','levels':['low','mid','high']}]})[0]['questions'][0]
        self.assertAlmostEqual(answer(q,[.1,.3,.6])['score'],1.5)
    def test_reject_duplicate_candidate_descriptions(self):
        self.request['questions'][0]['options']=['same','same']
        with self.assertRaises(ValueError):prepare(self.request)
    def test_reject_nonfinite_state_and_distribution(self):
        self.request['state']={'x':float('nan')}
        with self.assertRaises(ValueError):prepare(self.request)
        with self.assertRaises(RuntimeError):answer({'keys':['a','b']},[float('nan'),1.])
    def test_large_choice_is_supported_without_dropping(self):
        self.request['questions'][0]['options']=[f'action {i}' for i in range(255)]
        self.assertEqual(len(self.paths(self.request)),255)
    def test_batch_limit_is_enforced(self):
        self.request['questions'][0]['options']=[f'action {i}' for i in range(255)]
        with self.assertRaises(ValueError):prepare({'requests':[self.request]*5})

if __name__=='__main__':unittest.main()
