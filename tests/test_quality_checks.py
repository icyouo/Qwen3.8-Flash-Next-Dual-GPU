import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from q38_quality_checks import CASES,check_answer,validate_code

class QualityChecksTest(unittest.TestCase):
    def test_completion_and_exact_json_types(self):
        case=CASES[0]
        self.assertTrue(check_answer(case,'{"answer":391}',True)['pass'])
        self.assertFalse(check_answer(case,'{"answer":391}',False)['pass'])
        self.assertFalse(check_answer(case,'{"answer":391.0}',True)['pass'])
        self.assertFalse(check_answer(case,'{"answer":391,"extra":1}',True)['pass'])
        self.assertTrue(check_answer(case,'<think>work</think>{"answer":391}<|im_end|>',True)['pass'])
        self.assertFalse(check_answer(case,'<think>{"answer":391}',True)['pass'])
    def test_code_exec_and_wrong_algorithm(self):
        case=next(c for c in CASES if c['name']=='insertion_sort')
        self.assertTrue(check_answer(case,'```python\ndef solve(values: list[int]) -> list[int]:\n    return sorted(values)\n```',True)['pass'])
        self.assertFalse(check_answer(case,'def solve(values):\n    return values',True)['pass'])
    def test_terminal_markers_with_whitespace(self):
        for suffix in ('<|im_end|>\n', '<|endoftext|> \n',
                       '<|im_end|>\n<|endoftext|>\n'):
            self.assertTrue(check_answer(CASES[0],'{"answer":391}'+suffix,True)['pass'])
            self.assertFalse(check_answer(CASES[0],'{"answer":392}'+suffix,True)['pass'])
        self.assertFalse(check_answer(CASES[0],'{"answer":391}<|im_end|>extra',True)['pass'])
        case=next(c for c in CASES if c['name']=='insertion_sort')
        self.assertTrue(check_answer(case,'```python\ndef solve(values):\n    return sorted(values)\n```<|im_end|>\n',True)['pass'])
    def test_rejects_escape_and_top_level_side_effects(self):
        for code in ['import os\ndef solve(x): return x',
                     'def solve(x): return x.__class__',
                     'def solve(x): return open("/tmp/escape","w")',
                     'def solve(x): return eval(x)',
                     '@print\ndef solve(x): return x',
                     'print(1)\ndef solve(x): return x']:
            with self.assertRaises(ValueError): validate_code(code)
    def test_non_terminating_code_is_bounded(self):
        case={'code_tests':[{'args':[[]],'expected':[]}]}
        self.assertFalse(check_answer(case,'def solve(x):\n    while True: pass',True)['pass'])
if __name__=='__main__':unittest.main()
