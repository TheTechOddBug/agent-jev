import copy
import unittest
from .candidate_v8 import normalize_request


class NormalizeRequestTests(unittest.TestCase):
    def test_equivalent_evidence_and_no_input_mutation(self):
        a={'state':{'requirements':'Return 1','implementation':'def f(): return 1',
                    'observed_public_tests':{'passed':True}},
           'questions':[{'id':'complete','type':'boolean','question':'Complete?'}]}
        before=copy.deepcopy(a)
        b=copy.deepcopy(a)
        b['state']['observed_public_tests']={'passed':True,'checks':[{'name':'sample','passed':True}],
                                           'passed_checks':1,'total_checks':1}
        self.assertEqual(normalize_request(a),normalize_request(b))
        self.assertEqual(a,before)

    def test_invalid_state_still_rejected(self):
        for value in [None,3,True,'']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalize_request({'state':value,'questions':[{'type':'boolean','question':'Complete?'}]})

    def test_batch_and_failure_preservation(self):
        request={'state':{'requirements':'x','implementation':'y',
                         'observed_public_tests':{'passed':True,'checks':[{'passed':False,'error':'bad'}]}},
                 'questions':[{'type':'boolean','question':'Complete?'}]}
        result=normalize_request({'requests':[request]})
        self.assertIn('contradictory_summary',result['requests'][0]['state'])
        self.assertIn('bad',result['requests'][0]['state'])


if __name__=='__main__':
    unittest.main()
