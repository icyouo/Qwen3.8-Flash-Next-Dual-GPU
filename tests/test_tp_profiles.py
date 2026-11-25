from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from q38_tp_profiles import execution_profile, controlled_environment


class FixedMtp4ProfileTest(unittest.TestCase):
    def test_profiles_remove_ambient_tuning_and_keep_runtime_paths(self):
        inherited = {'PATH': '/bin', 'LD_LIBRARY_PATH': '/cuda/lib64',
                     'Q38_TE_DEBUG': '1', 'Q38_TP_MOE_FP32_SHARED_A': '1',
                     'Q38_CUDA_PROFILE_DECODE': '1'}
        for name in ('mtp4-baseline', 'mtp4-candidate', 'mtp4-route-candidate', 'mtp4-register-candidate', 'mtp4-native-fast'):
            profile = execution_profile(name)
            env = controlled_environment(profile, inherited)
            self.assertEqual(profile['options']['mtp_max_draft'], 4)
            self.assertFalse(profile['options']['tp_adaptive_mtp'])
            self.assertEqual(env['Q38_TP_GPU_HANDSHAKE'], str(int(name != 'mtp4-baseline')))
            self.assertNotIn('Q38_TE_DEBUG', env)
            self.assertNotIn('Q38_CUDA_PROFILE_DECODE', env)
            self.assertEqual(env['Q38_TP_MOE_FP32_SHARED_A'], '0')
            self.assertEqual(env['LD_LIBRARY_PATH'], '/cuda/lib64')
        self.assertEqual(inherited['Q38_TE_DEBUG'], '1')

    def test_bf16_profile_selects_precision_explicitly(self):
        old = execution_profile('mtp4-native-prefill-candidate')
        profile = execution_profile('mtp4-native-bf16-candidate')
        self.assertEqual(old['options'], profile['options'])
        self.assertNotIn('Q38_TP_RSAG_FP24', profile['environment'])
        self.assertNotIn('Q38_TP_F32_BF16_RSAG', profile['environment'])
        self.assertEqual(profile['environment']['Q38_TP_MOE_BF16_REDUCE'], '1')
        inherited = {'Q38_TP_MOE_BF16_REDUCE': '1', 'Q38_TP_RSAG_FP24': '1'}
        env = controlled_environment(old, inherited)
        self.assertNotIn('Q38_TP_MOE_BF16_REDUCE', env)
        self.assertNotIn('Q38_TP_RSAG_FP24', env)

    def test_gdn_qk_profile_changes_only_prefill_cache(self):
        old = execution_profile('mtp4-native-bf16-candidate')
        new = execution_profile('mtp4-native-bf16-gdn-qk-candidate')
        self.assertEqual(old['options'], new['options'])
        self.assertEqual({k: v for k, v in new['environment'].items()
                          if old['environment'].get(k) != v},
                         {'Q38_CUDA_GDN_PREFILL_NORMALIZED_QK': '1'})
        self.assertTrue(all(new['environment'][k] == v for k, v in old['environment'].items()))
        self.assertNotIn('Q38_CUDA_GDN_PREFILL_NORMALIZED_QK',
                         controlled_environment(old, {'Q38_CUDA_GDN_PREFILL_NORMALIZED_QK': '1'}))

    def test_unknown_profile_fails_closed(self):
        with self.assertRaises(ValueError):
            execution_profile('mtp6')

    def test_prefill_experiments_each_enable_one_flag(self):
        base_name = 'mtp4-native-bf16-gdn-qk-candidate'
        base = execution_profile(base_name)
        flags_by_profile = {
            'mtp4-native-bf16-qsa-compact-gate-candidate': 'Q38_CUDA_QSA_PREFILL_COMPACT_GATE',
            'mtp4-native-bf16-gdn-packed-qkvz-candidate': 'Q38_CUDA_GDN_PREFILL_PACKED_QKVZ',
            'mtp4-native-bf16-prefill-graph-candidate': 'Q38_TP_PREFILL_GRAPH',
            'mtp4-native-bf16-mtp-kv-only-candidate': 'Q38_CUDA_MTP_KV_ONLY',
            'mtp4-native-bf16-qsa-index-tile-candidate': 'Q38_CUDA_QSA_TILE_MTP_INDEX',
        }
        inherited = {flag: '1' for flag in flags_by_profile.values()}
        self.assertEqual(controlled_environment(base, inherited), base['environment'])
        for name, flag in flags_by_profile.items():
            with self.subTest(name=name):
                candidate = execution_profile(name)
                self.assertEqual(candidate['name'], name)
                self.assertEqual(candidate['options'], base['options'])
                self.assertEqual(controlled_environment(candidate, inherited),
                                 {**base['environment'], flag: '1'})

    def test_mtp_decode_experiment_keeps_bulk_prefill_graph_disabled(self):
        base = execution_profile('mtp4-native-bf16-gdn-qk-candidate')
        candidate = execution_profile('mtp4-native-bf16-mtp-decode-candidate')
        self.assertEqual(candidate['options'], base['options'])
        self.assertEqual(controlled_environment(candidate, {'Q38_TP_PREFILL_GRAPH': '1'}), {
            **base['environment'],
            'Q38_CUDA_MTP_KV_ONLY': '1',
            'Q38_CUDA_QSA_TILE_MTP_INDEX': '1',
        })

    def test_prefill_candidate_keeps_qualified_numerics_and_mtp_width(self):
        old = execution_profile('mtp4-native-longinput')
        new = execution_profile('mtp4-native-prefill-candidate')
        self.assertEqual(old['options'], new['options'])
        self.assertEqual({k: v for k, v in new['environment'].items()
                          if old['environment'].get(k) != v},
                         {'Q38_TP_BULK_COPY': '1'})
        env = controlled_environment(old, {'Q38_TP_BULK_COPY': '1'})
        self.assertNotIn('Q38_TP_BULK_COPY', env)

    def test_longinput_profile_changes_only_independent_work_partitioning(self):
        old = execution_profile('mtp4-native-fast')
        new = execution_profile('mtp4-native-longinput')
        self.assertEqual(old['options'], new['options'])
        added = {k: v for k, v in new['environment'].items() if k not in old['environment']}
        self.assertEqual(added, {'Q38_CUDA_QSA_SPARSE_SCORE_TILES': '8',
                                 'Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS': '16'})
        self.assertTrue(all(new['environment'][k] == v for k, v in old['environment'].items()))

    def test_routing_candidate_keeps_mtp4_and_isolates_flags(self):
        old = execution_profile('mtp4-candidate')
        new = execution_profile('mtp4-route-candidate')
        self.assertEqual(old['options'], new['options'])
        flags = {'Q38_CUDA_MOE_ROUTE_PARALLEL', 'Q38_CUDA_MOE_ROUTE_HASH_SKIP'}
        self.assertEqual({k for k in old['environment']
                          if old['environment'][k] != new['environment'][k]}, flags)
        for flag in flags:
            self.assertEqual(old['environment'][flag], '0')
            self.assertEqual(new['environment'][flag], '1')
        env = controlled_environment(old, {flag: '1' for flag in flags})
        for flag in flags:
            self.assertEqual(env[flag], '0')

    def test_register_candidate_isolated_and_fixed_mtp4(self):
        old = execution_profile('mtp4-route-candidate')
        new = execution_profile('mtp4-register-candidate')
        self.assertEqual(old['options'], new['options'])
        flags = {'Q38_CUDA_MOE_REG_MMA', 'Q38_TP_COMPACT_HEAD_BATCH'}
        self.assertEqual({k for k in old['environment']
            if old['environment'][k] != new['environment'][k]}, flags)
        self.assertEqual(new['options']['mtp_max_draft'], 4)
        for flag in flags:
            self.assertEqual(old['environment'][flag], '0')
            self.assertEqual(new['environment'][flag], '1')
        env = controlled_environment(old, {flag: '1' for flag in flags})
        self.assertTrue(all(env[flag] == '0' for flag in flags))

    def test_native_profile_rejects_ambient_precision_and_width_changes(self):
        profile = execution_profile('mtp4-native-fast')
        env = controlled_environment(profile, {
            'Q38_CUDA_HEAD_WARPS': '4', 'Q38_CUDA_MARLIN_MOE': '0',
            'Q38_TP_PEER_FUSED': '0', 'Q38_CUDA_GDN_PARTITION_COLUMNS': '32',
            'Q38_TP_ADAPTIVE_MTP': '1', 'Q38_CUDA_PROFILE_GRAPHS': '1'})
        self.assertEqual(env['Q38_CUDA_HEAD_WARPS'], '8')
        self.assertEqual(env['Q38_CUDA_MARLIN_MOE'], '1')
        self.assertEqual(env['Q38_TP_PEER_FUSED'], '1')
        self.assertEqual(env['Q38_CUDA_GDN_PARTITION_COLUMNS'], '16')
        self.assertNotIn('Q38_TP_ADAPTIVE_MTP', env)
        self.assertNotIn('Q38_CUDA_PROFILE_GRAPHS', env)
        self.assertFalse(profile['options']['tp_adaptive_mtp'])
        self.assertEqual(profile['options']['mtp_max_draft'], 4)

    def test_qualified_followup_keeps_failed_experiments_disabled(self):
        enabled = ('Q38_TP_MOE_SHARED_OVERLAP', 'Q38_CUDA_MARLIN_DIRECT_REDUCE',
                   'Q38_CUDA_MOE_HC_FINISH_FUSED', 'Q38_CUDA_MTP_INPUT_CACHE')
        disabled = ('Q38_CUDA_MARLIN_PERSISTENT', 'Q38_CUDA_MARLIN_WIDE_N',
                    'Q38_CUDA_GDN_CONV_PREPARE_FUSED')
        inherited = {key: '0' for key in enabled}
        inherited.update({key: '1' for key in disabled})
        profile = execution_profile('mtp4-native-fast')
        env = controlled_environment(profile, inherited)
        self.assertTrue(all(env[key] == '1' for key in enabled))
        self.assertTrue(all(env[key] == '0' for key in disabled))
        self.assertEqual(profile['options']['mtp_max_draft'], 4)
        self.assertFalse(profile['options']['tp_adaptive_mtp'])


if __name__ == '__main__':
    unittest.main()
