import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "q38_ple_index.py"
SPEC = importlib.util.spec_from_file_location("q38_ple_index", MODULE_PATH)
assert SPEC and SPEC.loader
ple_index = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ple_index
SPEC.loader.exec_module(ple_index)


class PleIndexTest(unittest.TestCase):
    def test_compile_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ple_dir = root / "ple"
            ple_dir.mkdir()
            data_path = ple_dir / "rows.bin"
            data_path.write_bytes(bytes(range(32)))
            manifest = {
                "storage_dtype": "BF16",
                "alignment_bytes": 2,
                "row_stride_bytes": 2,
                "embedding_row_dimension": 1,
                "usable_vocabulary_rows": 16,
                "padded_vocabulary_rows": 16,
                "layer_multipliers": [3, 5, 7],
                "per_head_vocabulary_sizes": [1] * 16,
                "per_head_offsets": list(range(16)),
                "physical_files": [
                    {
                        "index": 0,
                        "path": "ple/rows.bin",
                        "file_bytes": 32,
                        "payload_bytes": 32,
                    }
                ],
                "logical_parts": [
                    {
                        "logical_part": 0,
                        "physical_file_index": 0,
                        "global_row_start": 0,
                        "rows": 16,
                        "file_offset": 0,
                        "payload_bytes": 32,
                    }
                ],
            }
            manifest_path = ple_dir / "ple-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output_path = root / "layout.q38p"
            result = ple_index.compile_layout(manifest_path, output_path, 32, 31)
            self.assertEqual(result["parts"], 1)
            text = output_path.read_text()
            self.assertIn("Q38_PLE_LAYOUT_V1", text)
            self.assertIn(str(data_path.resolve()), text)


if __name__ == "__main__":
    unittest.main()
