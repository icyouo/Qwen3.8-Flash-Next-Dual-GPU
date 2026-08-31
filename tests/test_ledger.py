import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "q38_ledger.py"
SPEC = importlib.util.spec_from_file_location("q38_ledger", MODULE_PATH)
assert SPEC and SPEC.loader
ledger = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ledger
SPEC.loader.exec_module(ledger)


def write_string(file, value: str) -> None:
    encoded = value.encode()
    file.write(struct.pack("<Q", len(encoded)))
    file.write(encoded)


def write_metadata(file, key: str, value_type: int, value) -> None:
    write_string(file, key)
    file.write(struct.pack("<I", value_type))
    if value_type == ledger.VALUE_STRING:
        write_string(file, value)
    elif value_type == ledger.VALUE_UINT32:
        file.write(struct.pack("<I", value))
    elif value_type == ledger.VALUE_ARRAY:
        element_type, items = value
        file.write(struct.pack("<IQ", element_type, len(items)))
        for item in items:
            file.write(struct.pack("<I", item))
    else:
        raise AssertionError(value_type)


def make_tiny_gguf(path: Path) -> None:
    metadata = [
        ("general.architecture", ledger.VALUE_STRING, "qwen4exp"),
        ("general.alignment", ledger.VALUE_UINT32, 32),
        ("qwen4exp.block_count", ledger.VALUE_UINT32, 2),
        (
            "qwen4exp.attention.full_layers",
            ledger.VALUE_ARRAY,
            (ledger.VALUE_UINT32, [1]),
        ),
    ]
    tensors = [
        ("token_embd.weight", (4, 2), 30, 0),
        ("blk.0.ffn.weight", (2, 2), 0, 32),
        ("blk.1.ffn.weight", (32,), 8, 64),
        ("output.weight", (4, 2), 30, 128),
    ]
    with path.open("wb") as file:
        file.write(ledger.GGUF_MAGIC)
        file.write(struct.pack("<IQQ", ledger.GGUF_VERSION, len(tensors), len(metadata)))
        for item in metadata:
            write_metadata(file, *item)
        for name, dims, type_id, offset in tensors:
            write_string(file, name)
            file.write(struct.pack("<I", len(dims)))
            file.write(struct.pack("<" + "Q" * len(dims), *dims))
            file.write(struct.pack("<IQ", type_id, offset))
        data_start = ledger.align_up(file.tell(), 32)
        file.write(b"\0" * (data_start - file.tell()))
        file.write(b"\0" * 160)


class LedgerTest(unittest.TestCase):
    def test_qwen_owner_classification(self):
        self.assertEqual(ledger.classify_tensor("vblk.3.ffn_up.weight"), ("vision", None))
        self.assertEqual(ledger.classify_tensor("hc_input.mix_up.weight"), ("stage0_global", None))
        self.assertEqual(ledger.classify_tensor("mtp.blk.0.ffn.weight"), ("mtp", None))
        self.assertEqual(ledger.classify_tensor("blk.47.ffn.weight"), ("layer", 47))

    def test_parse_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            make_tiny_gguf(path)
            shard = ledger.parse_shard(path)
            self.assertEqual(len(shard.tensors), 4)
            self.assertEqual(shard.tensors[0].payload_bytes, 16)
            self.assertEqual(shard.tensors[2].payload_bytes, 34)
            report = ledger.build_ledger([shard], [1], 128)
            self.assertEqual(report["model"]["layer_count"], 2)
            self.assertEqual(report["candidate_plans"][0]["stage0_qsa_layers"], 0)
            self.assertEqual(report["candidate_plans"][0]["stage1_qsa_layers"], 1)
            self.assertEqual(report["unassigned_bytes"], 0)
            index_dir = Path(directory) / "indexes"
            paths = ledger.write_tensor_indexes(report, [shard], 1, index_dir)
            self.assertTrue(paths[0].is_file())
            self.assertIn("stage=0", paths[0].read_text())
            self.assertIn("blk.1.ffn.weight", paths[1].read_text())

    def test_rejects_tensor_extent_outside_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            make_tiny_gguf(path)
            path.write_bytes(path.read_bytes()[:-80])
            with self.assertRaises(ledger.GGUFError):
                ledger.parse_shard(path)


if __name__ == "__main__":
    unittest.main()
