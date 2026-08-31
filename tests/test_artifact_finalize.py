import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import q38_artifact_finalize as finalize


class ArtifactFinalizeTest(unittest.TestCase):
    def test_cut25_contract_matches_official_census(self):
        stage0 = finalize.expected_stage_tensor_names(0, 25)
        stage1 = finalize.expected_stage_tensor_names(1, 25)
        self.assertEqual(len(stage0), 738)
        self.assertEqual(len(stage1), 588)
        self.assertIn("token_embd.weight", stage0)
        self.assertIn("token_embd.weight", stage1)
        self.assertNotIn("hc_input.norm.weight", stage0)
        self.assertIn("hc_input.norm.weight", stage1)
        self.assertIn("ple.table.part.127", stage0)
        self.assertNotIn("ple.table.part.127", stage1)
        self.assertIn("mtp.blk.0.attn_q.weight", stage1)

    def test_contract_rejects_missing_or_unexpected_tensor(self):
        names = finalize.expected_stage_tensor_names(1, 25)
        finalize.validate_stage_tensor_contract(names, 1, 25)
        missing = set(names)
        missing.remove("mtp.fc_hidden.weight")
        with self.assertRaisesRegex(ValueError, "missing=mtp.fc_hidden.weight"):
            finalize.validate_stage_tensor_contract(missing, 1, 25)
        unexpected = set(names)
        unexpected.add("old.runtime.weight")
        with self.assertRaisesRegex(ValueError, "unexpected=old.runtime.weight"):
            finalize.validate_stage_tensor_contract(unexpected, 1, 25)


if __name__ == "__main__":
    unittest.main()
