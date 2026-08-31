import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import q38_ple_device_layout as compiler


class PleDeviceLayoutTest(unittest.TestCase):
    def test_layout_from_device_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment.q38w"
            data = bytearray(4096)
            struct.pack_into("<3q", data, 0, 3, 5, 7)
            struct.pack_into("<16q", data, 64, *([1] * 16))
            struct.pack_into("<16q", data, 256, *range(16))
            segment.write_bytes(data)
            digest = "a" * 64
            index = root / "index.q38d"
            records = [
                ("blk.1.ple.layer_multipliers", "I64", "preserve", 0, 24, "3"),
                ("blk.1.ple.head_vocab_sizes", "I64", "preserve", 64, 128, "16"),
                ("blk.1.ple.head_offsets", "I64", "preserve", 256, 128, "16"),
                ("ple.table.part.000", "BF16", "fp8_e4m3fn", 512, 1600, 3712, 20, "10,160"),
                ("ple.table.part.001", "BF16", "fp8_e4m3fn", 2112, 1600, 3732, 20, "10,160"),
            ]
            lines = ["Q38_DEVICE_INDEX_V1", "stage=0", "cut=25",
                     "source_repo=x", "source_commit=" + "c" * 40,
                     "policy_sha256=" + digest, "artifact_sha256=" + digest,
                     f"segment\t{segment.name}\t4096\t{digest}"]
            for record in records:
                if len(record) == 6:
                    name, dtype, fmt, offset, count, shape = record
                    scale_offset, scale_count = 0, 0
                else:
                    name, dtype, fmt, offset, count, scale_offset, scale_count, shape = record
                lines.append(
                    f"tensor\t{name}\tsource\t{dtype}\t{fmt}\t0\t0\t{offset}\t"
                    f"{count}\t{digest}\t{scale_offset}\t{scale_count}\t"
                    f"{digest if scale_count else '-'}\t{shape}"
                )
            index.write_text("\n".join(lines) + "\n")
            output = root / "ple.q38p"
            result = compiler.compile_layout(index, output)
            self.assertEqual(result["parts"], 2)
            self.assertEqual(result["usable_rows"], 16)
            self.assertIn("padded_rows=20", output.read_text())


if __name__ == "__main__":
    unittest.main()
