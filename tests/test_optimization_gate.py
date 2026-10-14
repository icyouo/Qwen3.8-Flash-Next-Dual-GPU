import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from q38_optimization_gate import CASES, measure, read_tokens
from q38_rpc_client import OPCODES


class FakeClient:
    def __init__(self):
        self.canonical = self.target = self.emitted = 0
        self.widths = []

    def call(self, opcode, **kwargs):
        output = []
        message = ""
        if opcode == OPCODES["append"]:
            self.canonical += len(kwargs["tokens"])
            self.target = self.canonical
        elif opcode == OPCODES["stats"]:
            message = json.dumps({"executor": {"failures": 0, "rollbacks": 0}})
        else:
            width = kwargs.get("argument1", 0)
            self.widths.append(width)
            count = width + 1
            output = list(range(100 + self.emitted, 100 + self.emitted + count))
            self.emitted += count
            self.canonical += count
            self.target = self.canonical - 1
        return SimpleNamespace(status=0, message=message, tokens=output,
                               frontiers=(self.canonical, self.target, self.target,
                                          self.target, self.target, 1))


class OptimizationGateTest(unittest.TestCase):
    def test_fixed_output_budget_excludes_seed_and_clamps_drafts(self):
        for budget in (2, 7, 9, 64):
            a, b = FakeClient(), FakeClient()
            plain = measure(a, [1, 2, 3], budget, 0, set())
            mtp = measure(b, [1, 2, 3], budget, 4, set())
            self.assertEqual(plain["output_tokens"], mtp["output_tokens"])
            self.assertEqual(mtp["measured_decode_tokens"], budget - 1)
            self.assertEqual(sum(mtp["tokens_per_transaction"]), budget - 1)
            self.assertLessEqual(max(b.widths), 4)

    def test_seed_stop_is_not_counted_as_decode(self):
        result = measure(FakeClient(), [1, 2], 64, 4, {100})
        self.assertEqual(result["output_tokens"], [100])
        self.assertEqual(result["measured_decode_tokens"], 0)
        self.assertEqual(result["sample_count"], 0)

    def test_invalid_fixtures_and_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.json"
            for fixture in ([True], [-1], [248320], [1.2], []):
                path.write_text(json.dumps(fixture))
                with self.assertRaises(ValueError):
                    read_tokens(path, 64)
            path.write_text(json.dumps({"tokens": [1, 2]}))
            self.assertEqual(read_tokens(path, 64), [1, 2])
            with self.assertRaises(ValueError):
                read_tokens(path, 262144)

    def test_new_lanes_are_separately_selectable(self):
        self.assertEqual(CASES["baseline"], ({}, {}))
        self.assertTrue(CASES["mtp-adaptive"][1]["enable_adaptive_mtp"])
        self.assertTrue(CASES["compact"][1]["enable_compact_sampling"])


if __name__ == "__main__":
    unittest.main()
