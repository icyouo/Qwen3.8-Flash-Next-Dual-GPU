import unittest
import re

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import q38_quant_policy as policy


class QuantPolicyTest(unittest.TestCase):
    def test_routed_expert_policy(self):
        edge = policy.decide(
            "model.language_model.layers.0.mlp.experts.gate_up_proj"
        )
        self.assertEqual(edge.format, policy.W8A16_G128)
        self.assertEqual(edge.canonical_name, "blk.0.ffn_gate_up_exps.weight")
        interior = policy.decide(
            "model.language_model.layers.25.mlp.experts.down_proj"
        )
        self.assertEqual(interior.format, policy.W4A16_G128)
        self.assertEqual(interior.owner, "stage1")

    def test_ple_and_vision(self):
        ple = policy.decide(
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_127.weight"
        )
        self.assertEqual(ple.format, policy.FP8_E4M3)
        self.assertEqual(ple.canonical_name, "ple.table.part.127")
        vision = policy.decide("model.visual.blocks.0.attn.qkv.weight")
        self.assertEqual(vision.format, policy.SKIP)
        for name in ("norm_conv", "norm_key", "norm_query"):
            norm = policy.decide(
                f"model.language_model.layers.1.ple.{name}.weight"
            )
            self.assertEqual(norm.format, policy.PRESERVE)

    def test_attention_and_mtp_names(self):
        q = policy.decide(
            "model.language_model.layers.3.self_attn.q_proj.weight"
        )
        self.assertEqual(q.canonical_name, "blk.3.attn_q.weight")
        self.assertEqual(q.format, policy.W8A16_G128)
        mtp = policy.decide("mtp.layers.0.mlp.experts.down_proj")
        self.assertEqual(mtp.owner, "stage1")
        self.assertEqual(mtp.format, policy.W8A16_G128)
        index_norm = policy.decide(
            "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight"
        )
        self.assertEqual(index_norm.format, policy.PRESERVE)
        for name in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
            self.assertEqual(policy.decide(name).format, policy.PRESERVE)

    def test_policy_identity_stable(self):
        self.assertEqual(len(policy.policy_identity()), 64)
        self.assertEqual(policy.policy_identity(), policy.policy_identity())
        header = (
            Path(__file__).resolve().parents[1]
            / "include"
            / "q38"
            / "production_contract.h"
        ).read_text()
        match = re.search(
            r'kQ38AmperePolicySha256\[\]\s*=\s*\n?\s*"([0-9a-f]{64})"',
            header,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), policy.policy_identity(25))

    def test_final_hyper_mixer_belongs_to_stage1(self):
        decision = policy.decide(
            "model.language_model.hyper_connection_mixer.hc_norm.weight", 25
        )
        self.assertEqual(decision.owner, "stage1")
        self.assertEqual(decision.canonical_name, "hc_input.norm.weight")


if __name__ == "__main__":
    unittest.main()
