import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from q38_safetensors import read_safetensors_header


class SafeTensorsTest(unittest.TestCase):
    def write_fixture(self, root: Path, payload: bytes = bytes(range(16))) -> Path:
        header = {
            "a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
            "b": {"dtype": "I64", "shape": [1], "data_offsets": [8, 16]},
            "__metadata__": {"format": "pt"},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = root / "fixture.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def test_reads_header_and_extents(self):
        with tempfile.TemporaryDirectory() as directory:
            result = read_safetensors_header(self.write_fixture(Path(directory)))
            self.assertEqual(result.metadata, {"format": "pt"})
            self.assertEqual([item.name for item in result.tensors], ["a", "b"])
            self.assertEqual(result.tensors[0].shape, (2, 2))
            self.assertEqual(result.tensors[1].nbytes, 8)
            self.assertEqual(result.tensors[0].absolute_begin, result.data_begin)

    def test_rejects_partial_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), bytes(range(8)))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_safetensors_header(path)
            result = read_safetensors_header(path, verify_payload_extents=False)
            self.assertEqual(len(result.tensors), 2)

    def test_rejects_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            header = {
                "a": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]}
            }
            encoded = json.dumps(header).encode()
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"xxx")
            with self.assertRaisesRegex(ValueError, "gap"):
                read_safetensors_header(path)


if __name__ == "__main__":
    unittest.main()
