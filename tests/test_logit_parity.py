from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_logit_parity import compare_captures, fnv1a64, write_capture  # noqa: E402


def snapshot(raw: bytes, selected_token: int = 3) -> dict[str, object]:
    return {
        "schema": "q38.logits.bf16.v1",
        "dtype": "bf16",
        "byte_order": "little",
        "transaction_epoch": 7,
        "target_frontier": 42,
        "transaction_kind": "decode",
        "selected_token": selected_token,
        "element_count": len(raw) // 2,
        "fnv1a64": fnv1a64(raw),
        "raw_bf16_le_hex": raw.hex(),
    }


class LogitParityTest(unittest.TestCase):
    def test_exact_capture_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b"\x00\x3f\x80\x3f\x00\xc0"
            write_capture(snapshot(raw), root / "eager")
            write_capture(snapshot(raw), root / "graph")
            result = compare_captures(root / "eager.json", root / "graph.json")
            self.assertTrue(result["parity_pass"])
            self.assertTrue(result["exact_bf16_match"])
            self.assertEqual(result["mismatched_elements"], 0)
            self.assertIsNone(result["first_mismatch"])

    def test_reports_first_bf16_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_capture(snapshot(b"\x00\x3f\x80\x3f"), root / "eager")
            write_capture(snapshot(b"\x00\x3f\x81\x3f", 4), root / "graph")
            result = compare_captures(root / "eager.json", root / "graph.json")
            self.assertFalse(result["parity_pass"])
            self.assertFalse(result["exact_bf16_match"])
            self.assertEqual(result["mismatched_elements"], 1)
            self.assertEqual(
                result["first_mismatch"],
                {"index": 1, "left_bf16_bits": 0x3F80, "right_bf16_bits": 0x3F81},
            )

    def test_rejects_corrupted_raw_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_capture(snapshot(b"\x00\x3f"), root / "capture")
            (root / "capture.bf16").write_bytes(b"\x01\x3f")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                compare_captures(root / "capture.json", root / "capture.json")


if __name__ == "__main__":
    unittest.main()
