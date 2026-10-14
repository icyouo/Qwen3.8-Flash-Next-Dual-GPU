"""Deterministic answer checks for native/reference model qualification.

Code answers run in a separate, bounded Python process after restrictive AST
validation. This is a small algorithm fixture runner, not a general code service.
"""
from __future__ import annotations
import ast
import json
import re
import subprocess
import sys


CASES = [
    {'name': 'multiply', 'prompt': '计算17乘以23。只输出JSON对象，格式为{"answer":整数}。', 'expected': {'answer': 391}},
    {'name': 'fraction', 'prompt': '计算3/4加5/6，约分为最简分数。只输出JSON对象，格式为{"numerator":整数,"denominator":整数}。', 'expected': {'numerator': 19, 'denominator': 12}},
    {'name': 'linear', 'prompt': '求解3x+7=52。只输出JSON对象，格式为{"x":整数}。', 'expected': {'x': 15}},
    {'name': 'sort_json', 'prompt': '把[7,-2,7,0,13,-9]按升序排序，保留重复项。只输出JSON数组。', 'expected': [-9, -2, 0, 7, 7, 13]},
    {'name': 'copy_unicode', 'prompt': '逐字复制字符串蓝天白云_170HX，不添加任何字符。', 'expected_text': '蓝天白云_170HX'},
    {'name': 'extract', 'prompt': '从“甲买了3个苹果，乙买了5个苹果，甲又买了2个苹果”统计每人苹果总数。只输出JSON对象，用甲、乙作为键。', 'expected': {'甲': 5, '乙': 5}},
    {'name': 'insertion_sort', 'prompt': '仅输出一个Python代码块，定义函数solve(values)，用插入排序返回升序列表并保留重复项。不使用import。', 'code_tests': [
        {'args': [[]], 'expected': []}, {'args': [[3, 1, 2, 1]], 'expected': [1, 1, 2, 3]},
        {'args': [[0, -5, 9, -5]], 'expected': [-5, -5, 0, 9]}, {'args': [[7]], 'expected': [7]}]},
    {'name': 'binary_search', 'prompt': '仅输出一个Python代码块，定义solve(values,target)。values是升序且无重复的整数列表，用二分查找返回target的下标，不存在则返回-1。不使用import。', 'code_tests': [
        {'args': [[], 3], 'expected': -1}, {'args': [[-5, 0, 4, 9], -5], 'expected': 0},
        {'args': [[-5, 0, 4, 9], 9], 'expected': 3}, {'args': [[-5, 0, 4, 9], 3], 'expected': -1}]},
    {'name': 'deduplicate', 'prompt': '仅输出一个Python代码块，定义solve(values)，去除整数列表中的重复值但保留首次出现的顺序，返回列表。不使用import。', 'code_tests': [
        {'args': [[]], 'expected': []}, {'args': [[3, 1, 3, 2, 1]], 'expected': [3, 1, 2]},
        {'args': [[0, -1, 0, -1, 2]], 'expected': [0, -1, 2]}]},
]


def clean_answer(text: str) -> str:
    # Keep only content after a completed thinking block, never accept an
    # unfinished reasoning answer as a completed response.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        raise ValueError('unfinished thinking block')
    # Tokenizer decoding may append whitespace after a terminal marker. Strip
    # only terminal framing, never markers embedded in the answer itself.
    text = text.strip()
    while True:
        for marker in ('<|im_end|>', '<|endoftext|>'):
            if text.endswith(marker):
                text = text[:-len(marker)].rstrip()
                break
        else:
            return text


def code_from_answer(text: str) -> str:
    match = re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```', text, re.S)
    return match.group(1) if match else text


def validate_code(code: str) -> ast.Module:
    tree = ast.parse(code)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError('only function definitions are allowed')
    for function in tree.body:
        if function.decorator_list:
            raise ValueError('decorators are unsupported')
        function.returns = None
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs,
                         function.args.vararg, function.args.kwarg]:
            if argument is not None: argument.annotation = None
    names = {n.name for n in tree.body}
    if 'solve' not in names:
        raise ValueError('solve function missing')
    builtins = {'range', 'len', 'sorted', 'sum', 'min', 'max', 'abs', 'enumerate', 'zip', 'list', 'dict', 'set', 'int', 'str', 'bool', 'reversed'}
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith,
                 ast.Global, ast.Nonlocal, ast.Try, ast.Raise, ast.AsyncFunctionDef,
                 ast.Await, ast.Yield, ast.YieldFrom)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError('unsupported statement')
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name.startswith('__'):
                raise ValueError('dunder names are not allowed')
        if isinstance(node, ast.Attribute) and node.attr not in ('append', 'add', 'copy'):
            raise ValueError('unsupported attribute')
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in builtins | names or
                isinstance(node.func, ast.Attribute) and node.func.attr in ('append', 'add', 'copy')):
            raise ValueError('unsupported call')
    return tree


def exact_value(actual, expected) -> bool:
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def check_answer(case: dict, text: str, completed: bool) -> dict:
    if not completed:
        return {'pass': False, 'reason': 'generation did not finish'}
    try:
        text = clean_answer(text)
        if 'expected_text' in case:
            actual = text
            expected = case['expected_text']
        elif 'expected' in case:
            match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
            actual = json.loads(match.group(1) if match else text)
            expected = case['expected']
        else:
            code = code_from_answer(text)
            validate_code(code)
            result = subprocess.run([sys.executable, '-I', __file__, '--worker'],
                input=json.dumps({'code': code, 'tests': case['code_tests']}),
                capture_output=True, text=True, timeout=4)
            if result.returncode:
                return {'pass': False, 'reason': 'code execution failed', 'returncode': result.returncode,
                        'stderr': result.stderr[-1000:]}
            actual = json.loads(result.stdout)
            expected = [t['expected'] for t in case['code_tests']]
        return {'pass': exact_value(actual, expected), 'actual': actual, 'expected': expected}
    except (ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
        return {'pass': False, 'reason': str(exc)}


def worker() -> None:
    import builtins
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    if sys.platform.startswith('linux'):
        resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    payload = json.load(sys.stdin)
    tree = validate_code(payload['code'])
    allowed = ('range', 'len', 'sorted', 'sum', 'min', 'max', 'abs', 'enumerate', 'zip', 'list', 'dict', 'set', 'int', 'str', 'bool', 'reversed')
    scope = {'__builtins__': {name: getattr(builtins, name) for name in allowed}}
    exec(compile(tree, '<model-answer>', 'exec'), scope)
    answers = [scope['solve'](*test['args']) for test in payload['tests']]
    print(json.dumps(answers, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Use check_answer from the benchmark harness')
    worker()
