"""Explicit fixed-MTP4 execution policies; no ambient Q38 tuning overrides."""
from __future__ import annotations


def execution_profile(name: str) -> dict:
    if name == 'mtp4-native-bf16-gdn-qk-candidate':
        profile = execution_profile('mtp4-native-bf16-candidate')
        profile['name'] = name
        profile['environment']['Q38_CUDA_GDN_PREFILL_NORMALIZED_QK'] = '1'
        return profile
    if name == 'mtp4-native-bf16-candidate':
        profile = execution_profile('mtp4-native-prefill-candidate')
        profile['name'] = name
        profile['environment']['Q38_TP_MOE_BF16_REDUCE'] = '1'
        return profile
    if name == 'mtp4-native-prefill-fp24-candidate':
        profile = execution_profile('mtp4-native-prefill-rsag-candidate')
        profile['name'] = name
        profile['environment']['Q38_TP_RSAG_FP24'] = '1'
        return profile
    if name == 'mtp4-native-prefill-rsag-candidate':
        profile = execution_profile('mtp4-native-prefill-candidate')
        profile['name'] = name
        profile['environment']['Q38_TP_F32_BF16_RSAG'] = '1'
        return profile
    if name == 'mtp4-native-prefill-candidate':
        profile = execution_profile('mtp4-native-longinput')
        profile['name'] = name
        profile['environment']['Q38_TP_BULK_COPY'] = '1'
        return profile
    if name == 'mtp4-native-longinput':
        profile = execution_profile('mtp4-native-fast')
        profile['name'] = name
        profile['environment'].update(Q38_CUDA_QSA_SPARSE_SCORE_TILES='8',
                                      Q38_CUDA_GDN_PREFILL_PARTITION_COLUMNS='16')
        return profile
    if name not in ('mtp4-baseline', 'mtp4-candidate', 'mtp4-route-candidate', 'mtp4-register-candidate', 'mtp4-native-fast'):
        raise ValueError('unknown fixed-MTP4 execution profile: ' + name)
    candidate = name != 'mtp4-baseline'
    options = dict(tp_transport='p2p', tp_p2p_async=True,
                   tp_persistent_worker=True, tp_moe_prefill='grouped-fp32',
                   tp_small_gemm=True, tp_batched_target=True,
                   tp_prefix_snapshots=True, tp_piecewise_graph=True,
                   tp_batched_graph=True, tp_batched_mtp=True,
                   tp_adaptive_mtp=False, mtp_max_draft=4)
    enabled = ['Q38_TP_PERSISTENT_WORKER', 'Q38_TP_GROUPED_MOE',
               'Q38_TP_MOE_FP32_SCALES', 'Q38_TP_SMALL_GEMM',
               'Q38_TP_BATCHED_TARGET', 'Q38_TP_PREFIX_SNAPSHOTS',
               'Q38_TP_PIECEWISE_GRAPH', 'Q38_TP_BATCHED_GRAPH']
    changes = ['Q38_TP_PAIRED_GRAPH', 'Q38_TP_GPU_HANDSHAKE', 'Q38_TP_DEVICE_BOUNDARY', 'Q38_TP_RECOVER_CANCEL',
               'Q38_CUDA_MTP_GRAPH', 'Q38_CUDA_PLE_COPY_STREAM',
               'Q38_TP_HEAD_BATCH_REDUCE', 'Q38_TP_DEFER_HEAD_SYNC',
               'Q38_TP_MTP_KV_ONLY_COMMIT', 'Q38_CUDA_HC_FUSED',
               'Q38_CUDA_PLE_FUSED', 'Q38_CUDA_FUSED_PROJECTIONS']
    env = {key: '1' for key in enabled}
    env.update({key: str(int(candidate)) for key in changes})
    env.update({key: str(int(name in ('mtp4-route-candidate','mtp4-register-candidate','mtp4-native-fast'))) for key in
                ('Q38_CUDA_MOE_ROUTE_PARALLEL', 'Q38_CUDA_MOE_ROUTE_HASH_SKIP')})
    env.update({key: str(int(name in ('mtp4-register-candidate','mtp4-native-fast'))) for key in
                ('Q38_CUDA_MOE_REG_MMA', 'Q38_TP_COMPACT_HEAD_BATCH')})
    env.update({key: '0' for key in ['Q38_TP_MOE_FP32_SHARED_A',
        'Q38_TP_MOE_VALIDATE', 'Q38_MTP_DRAFT_DIAGNOSTICS',
        'Q38_TP_PREFIX_DIAGNOSTICS', 'Q38_TP_BATCH_GRAPH_DIAGNOSTICS']})
    if name == 'mtp4-native-fast':
        # Lossless vocabulary-row/QKV-Z artifact layouts and pinned Marlin MoE.
        # MTP width and canonical verification are shared with older profiles.
        env.update({key: '1' for key in (
            'Q38_CUDA_MARLIN_MOE', 'Q38_CUDA_TOPK_WARP', 'Q38_TP_VOCAB_ROWS',
            'Q38_TP_GDN_LOCAL', 'Q38_TP_GDN_PACKED', 'Q38_CUDA_HC_MIX_TENSOR',
            'Q38_CUDA_MARLIN_COMPACT_PLAN', 'Q38_CUDA_SMALL_GEMM_BORROWED',
            'Q38_TP_WHOLE_TARGET_GRAPH', 'Q38_TP_QSA_LOCAL',
            'Q38_TP_QSA_LOCAL_PROJECTION', 'Q38_CUDA_QSA_SMALL_UNGROUPED',
            'Q38_TP_WHOLE_MTP_GRAPH', 'Q38_TP_HEAD_PARALLEL_ARGMAX',
            'Q38_CUDA_MARLIN_FUSED_TAIL', 'Q38_CUDA_HC_FINISH_FUSED',
            'Q38_TP_PEER_FUSED', 'Q38_TP_MOE_SHARED_OVERLAP',
            'Q38_CUDA_MARLIN_DIRECT_REDUCE', 'Q38_CUDA_MOE_HC_FINISH_FUSED',
            'Q38_CUDA_MTP_INPUT_CACHE')})
        env.update({key: '0' for key in (
            'Q38_CUDA_HC_INJECT_BATCH', 'Q38_TP_PEER_VECTOR',
            'Q38_TP_HEAD_HYPER_BATCH',
            'Q38_TP_HEAD_GRAPH', 'Q38_CUDA_MARLIN_PERSISTENT',
            'Q38_CUDA_MARLIN_WIDE_N', 'Q38_CUDA_GDN_CONV_PREPARE_FUSED')})
        env.update(Q38_CUDA_HEAD_PIPELINED='20', Q38_CUDA_HEAD_WARPS='8',
                   Q38_CUDA_QSA_VALUE_STAGING='8',
                   Q38_CUDA_GDN_PARTITION_COLUMNS='16',
                   Q38_CUDA_QSA_SCORE_TILES='1', Q38_CUDA_QSA_VALUE_TILES='1')
    return dict(schema='Q38_FIXED_MTP4_PROFILE_V1', name=name,
                options=options, environment=env)


def controlled_environment(profile: dict, inherited: dict) -> dict:
    result = {key: value for key, value in inherited.items() if not key.startswith('Q38_')}
    result.update(profile['environment'])
    return result
