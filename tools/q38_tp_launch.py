#!/usr/bin/env python3
"""Launch the TP2/EP runtime with one transactional writer.

Builds a distinct session identity from all four rank indexes and the current
runtime. PP READY and its identity are read-only. Fixed MTP4 profiles select
batched verification and graph execution; manual mode preserves eager defaults.
Durable recovery remains disabled.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys

from q38_identity import canonical_sampling, fnv64, hash_group, hash_path, hash_runtime
from q38_launch import (PINNED_COMMIT, PINNED_REPO, READY_SCHEMA, parse_identity,
                        verify_segments, resolve_artifact_path, sampling_arguments,
                        sha256_file, terminate, wait_for_executor)
from q38_shard_plan import load_stage
from q38_tp_profiles import execution_profile, controlled_environment

ROOT = Path(__file__).resolve().parents[1]


def prepare(ready_path: Path, tp_root: Path, state: Path, verify_hashes: bool,
            moe_prefill: str = 'row', small_gemm: bool = False,
            segment_hash_workers: int = 1, batched_target: bool = False,
            prefix_snapshots: bool = False, piecewise_graph: bool = False,
            batched_graph: bool = False, batched_mtp: bool = False, adaptive_mtp: bool = False,
            execution_policy: dict | None = None) -> dict:
    if moe_prefill not in ('row','grouped','grouped-fp32'):
        raise ValueError('unknown TP MoE prefill policy')
    ready_path, tp_root = ready_path.resolve(), tp_root.resolve()
    ready = json.loads(ready_path.read_text())
    if (ready.get('schema'), ready.get('source_repo'), ready.get('source_commit')) != (
            READY_SCHEMA, PINNED_REPO, PINNED_COMMIT):
        raise ValueError('source READY provenance differs')
    paths = {name: resolve_artifact_path(ready_path, ready[name]) for name in (
        'stage0_index','stage1_index','ple_layout','session_identity','model_metadata',
        'chat_template','source_model_root')}
    for name, digest in [('stage0_index','stage0_index_sha256'),
                         ('stage1_index','stage1_index_sha256'),
                         ('ple_layout','ple_layout_sha256'),('session_identity','identity_sha256')]:
        if sha256_file(paths[name]) != ready[digest]:
            raise ValueError('source READY hash differs: '+name)
    prior = parse_identity(paths['session_identity'])
    manifest = json.loads((tp_root/'variant_manifest.json').read_text())
    if manifest.get('schema') != 'Q38_TP2_REPLICATED_CONTROLS_V1':
        raise ValueError('unrecognized TP2 variant manifest')
    indexes = []
    records = {(v['rank'],v['stage']): v for v in manifest['stages']}
    for rank in range(2):
        for stage in range(2):
            index = tp_root/f'rank{rank}/stage{stage}/index.q38d'
            if sha256_file(index) != records[(rank,stage)]['index_sha256']:
                raise ValueError('TP2 index hash differs: '+str(index))
            layout = load_stage(index)
            source = load_stage(paths[f'stage{stage}_index'])
            if (layout.stage,layout.cut,layout.source_commit,layout.source_repo,layout.policy_sha256) != (
                    stage,source.cut,source.source_commit,source.source_repo,source.policy_sha256):
                raise ValueError('TP2 index provenance differs')
            indexes.append(index)
    census = verify_segments(indexes, verify_hashes, segment_hash_workers)
    sampling = json.dumps(ready['sampling'])
    digests = [hash_path(paths['model_metadata']),
        hash_group([resolve_artifact_path(ready_path,p) for p in ready['tokenizer_files']]),
        hash_path(paths['chat_template']), hash_runtime(ROOT), hash_path(ROOT/'cuda'),
        hash_path(paths['ple_layout']), hash_group(indexes),
        hashlib.sha256(canonical_sampling(sampling)).digest()]
    # Grouped MMQ changes the floating-point arithmetic order. Bind that
    # explicit policy to identity; schedule-only P1 switches do not need one.
    if moe_prefill != 'row':
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_GROUPED_MOE_V1').digest()
    if moe_prefill == 'grouped-fp32':
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_MOE_FP32_GROUP_SCALES_V1').digest()
    if small_gemm:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_PADDED_SMALL_GEMM_V1').digest()
    if adaptive_mtp and not batched_mtp:
        raise ValueError('TP adaptive MTP requires batched MTP')
    if adaptive_mtp:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_BATCH_ADAPTIVE_MTP_V1').digest()
    if batched_mtp and not batched_target:
        raise ValueError('TP MTP requires batched target verification')
    if batched_mtp:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_BATCHED_MTP_V1').digest()
    if batched_graph and not (piecewise_graph and batched_target):
        raise ValueError('batched graphs require TP piecewise graph and batched target')
    if batched_graph:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_BATCHED_GRAPH_V1').digest()
    if piecewise_graph:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_PIECEWISE_GRAPH_V1').digest()
    if prefix_snapshots and not batched_target:
        raise ValueError('prefix snapshots require batched target verification')
    if prefix_snapshots:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_PREFIX_SNAPSHOTS_V1').digest()
    if batched_target:
        digests[4] = hashlib.sha256(digests[4] + b'Q38_TP_BATCHED_TARGET_V1').digest()
    if execution_policy is not None:
        digests[4] = hashlib.sha256(digests[4] +
            json.dumps(execution_policy, sort_keys=True, separators=(',', ':')).encode()).digest()
    labels=['model_checkpoint','tokenizer','chat_template','runtime','kernels','ple_layout','stage_plan','sampling_parser']
    for i in (0,1,2,5,7):
        if digests[i].hex() != prior[labels[i]+'_sha256']:
            raise ValueError('source identity input differs: '+labels[i])
    context=int(ready['context_limit'])
    # The identity binds both ranks, unlike the old PP identity used by smokes.
    session=int.from_bytes(hashlib.sha256(b'Q38_TP2_EAGER_V1'+b''.join(digests)).digest()[:8],'little') or 1
    raw=struct.pack('<IHHQ',int.from_bytes(b'Q38I','little'),1,288,session)+b''.join(digests)+struct.pack('<IIQ',context,0,0)
    identity=state/'session.q38i'
    lines=['Q38_SESSION_IDENTITY_V1',f'session_hash={session}']
    lines += [name+'_sha256='+digest.hex() for name,digest in zip(labels,digests)]
    lines += [f'context_limit={context}','flags=0',f'identity_checksum={fnv64(raw)}']
    temp=identity.with_suffix('.q38i.tmp');temp.write_text('\n'.join(lines)+'\n');os.replace(temp,identity)
    parse_identity(identity)
    result={'schema':'Q38_TP2_LAUNCH_V1','experimental':True,'durability':'off',
        'tp_root':str(tp_root),'session_hash':session,'identity':str(identity),
        'ple_layout':str(paths['ple_layout']),'tokenizer':str(paths['source_model_root']),
        'context_limit':context,'sampling':ready['sampling'],
        'verified_segment_hashes':verify_hashes,'unique_segments':census['segments'],'segment_bytes':census['segment_bytes'],
        'segment_hash_workers':segment_hash_workers}
    result['tp_moe_prefill']=moe_prefill
    result['tp_small_gemm']=small_gemm
    result['tp_adaptive_mtp']=adaptive_mtp
    result['mtp_enabled']=batched_mtp
    result['tp_batched_mtp']=batched_mtp
    result['tp_batched_target']=batched_target
    result['tp_prefix_snapshots']=prefix_snapshots
    result['tp_piecewise_graph']=piecewise_graph
    result['tp_batched_graph']=batched_graph
    (state/'runtime-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ready',type=Path,required=True)
    p.add_argument('--tp-root',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=ROOT/'build/te-repair/q38-cuda-runtime')
    p.add_argument('--port',type=int,default=30001)
    p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--tp-transport',choices=('host','p2p'),default='host')
    p.add_argument('--tp-p2p-async',action='store_true',help='enqueue P2P with stream dependencies instead of waiting on the CPU')
    p.add_argument('--tp-persistent-worker',action='store_true',help='reuse the peer rank thread per logical stage')
    p.add_argument('--tp-moe-prefill',choices=('row','grouped','grouped-fp32'),default='row')
    p.add_argument('--tp-adaptive-mtp',action='store_true',help='measure batch cost per published token to choose width')
    p.add_argument('--tp-batched-mtp',action='store_true',help='experimental paired draft and batched target MTP')
    p.add_argument('--mtp-max-draft',type=int,default=4)
    p.add_argument('--execution-profile',choices=('manual','mtp4-baseline','mtp4-candidate','mtp4-route-candidate','mtp4-register-candidate','mtp4-native-fast','mtp4-native-longinput','mtp4-native-prefill-candidate','mtp4-native-prefill-rsag-candidate','mtp4-native-prefill-fp24-candidate','mtp4-native-bf16-candidate','mtp4-native-bf16-gdn-qk-candidate'),default='manual')
    p.add_argument('--tp-batched-graph',action='store_true',help='experimental 2/3/5/7-row target graph fragments')
    p.add_argument('--tp-piecewise-graph',action='store_true',help='experimental TP GPU graph fragments with eager collectives')
    p.add_argument('--tp-prefix-snapshots',action='store_true',help='capture evaluated prefix states for target batches up to 7 rows')
    p.add_argument('--tp-batched-target',action='store_true',help='experimental multi-row candidate target verification')
    p.add_argument('--tp-small-gemm',action='store_true',help='pad 2..7 dense rows to an 8-row GEMM')
    p.add_argument('--ple-storage',choices=('ssd','ram'),default='ssd')
    p.add_argument('--ple-cache-gib',type=int,default=8)
    p.add_argument('--ple-ram-budget-gib',type=int,default=64)
    p.add_argument('--skip-segment-hashes',action='store_true',help='diagnostic only: verify indexes/extents but skip full segment hashes')
    p.add_argument('--segment-hash-workers',type=int,choices=range(1,17),default=1)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    profile = None
    if a.execution_profile != 'manual':
        if a.tp_adaptive_mtp or a.mtp_max_draft != 4:
            p.error('fixed MTP4 profiles require width4 and disable adaptive MTP')
        profile = execution_profile(a.execution_profile)
        for key, value in profile['options'].items(): setattr(a, key, value)
    if not 1 <= a.port <= 65535 or not 1 <= a.chunk <= 512: p.error('invalid port or chunk')
    if not 1 <= a.mtp_max_draft <= 64: p.error('invalid MTP draft width')
    if a.tp_batched_mtp and a.mtp_max_draft + 1 > a.chunk: p.error('MTP width exceeds target batch capacity')
    if a.tp_p2p_async and a.tp_transport!='p2p': p.error('--tp-p2p-async requires --tp-transport p2p')
    if not 1 <= a.ple_cache_gib <= (2**64-1) >> 30 or not 1 <= a.ple_ram_budget_gib <= (2**64-1) >> 30:
        p.error('invalid PLE cache or RAM budget')
    state=a.state_dir.resolve();state.mkdir(parents=True,exist_ok=True)
    lock=(state/'launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(json.dumps({'event':'verify','tp_root':str(a.tp_root),'full_segment_hashes':not a.skip_segment_hashes}),flush=True)
    runtime_sha = sha256_file(a.runtime.resolve()) if a.runtime.exists() else None
    if profile and runtime_sha is None: p.error('fixed MTP4 profile requires a built runtime')
    policy = dict(profile=profile, runtime_sha256=runtime_sha) if profile else None
    info=prepare(a.ready,a.tp_root,state,not a.skip_segment_hashes,a.tp_moe_prefill,a.tp_small_gemm,a.segment_hash_workers,a.tp_batched_target,a.tp_prefix_snapshots,a.tp_piecewise_graph,a.tp_batched_graph,a.tp_batched_mtp,a.tp_adaptive_mtp,policy)
    info['tp_transport']=a.tp_transport
    info['tp_stream_ordered']=a.tp_p2p_async
    info['tp_persistent_worker']=a.tp_persistent_worker
    info['ple_storage']=a.ple_storage
    info['ple_ram_budget_gib']=a.ple_ram_budget_gib
    info['execution_profile']=profile
    info['runtime_sha256']=runtime_sha
    (state/'runtime-manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    socket=state/'executor.sock'
    command=[str(a.runtime.resolve()),'--tp2-root',info['tp_root'],'--ple-layout',info['ple_layout'],
        '--identity',info['identity'],'--session-hash',str(info['session_hash']),
        '--socket',str(socket),'--context-limit',str(info['context_limit']),'--chunk',str(a.chunk),
        '--ple-io','ram' if a.ple_storage=='ram' else 'direct',
        '--ple-cache-gib',str(a.ple_cache_gib),
        *sampling_arguments(info['sampling'])]
    if a.ple_storage=='ram': command.extend(['--ple-ram-budget-gib',str(a.ple_ram_budget_gib)])
    if a.tp_batched_mtp: command.append('--tp-batched-mtp')
    if a.tp_adaptive_mtp: command.append('--enable-adaptive-mtp')
    if a.tp_transport=='p2p': command.append('--tp-p2p')
    if a.tp_p2p_async: command.append('--tp-p2p-async')
    (state/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    if a.prepare_only:
        print(json.dumps(info),flush=True);return 0
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    executor=None;sidecar=None
    try:
        environment=os.environ.copy()
        environment['Q38_TP_PERSISTENT_WORKER']='1' if a.tp_persistent_worker else '0'
        environment['Q38_TP_GROUPED_MOE']='1' if a.tp_moe_prefill!='row' else '0'
        environment['Q38_TP_MOE_FP32_SCALES']='1' if a.tp_moe_prefill=='grouped-fp32' else '0'
        environment['Q38_TP_BATCHED_TARGET']='1' if a.tp_batched_target else '0'
        environment['Q38_TP_PREFIX_SNAPSHOTS']='1' if a.tp_prefix_snapshots else '0'
        environment['Q38_TP_PIECEWISE_GRAPH']='1' if a.tp_piecewise_graph else '0'
        environment['Q38_TP_BATCHED_GRAPH']='1' if a.tp_batched_graph else '0'
        environment['Q38_TP_SMALL_GEMM']='1' if a.tp_small_gemm else '0'
        if profile:
            environment = controlled_environment(profile, environment)
        executor=subprocess.Popen(command,env=environment)
        wait_for_executor(socket,info['session_hash'],executor,
                          timeout=900 if a.ple_storage=='ram' else 300)
        sidecar=subprocess.Popen([sys.executable,str(ROOT/'tools/q38_sidecar.py'),
            '--socket',str(socket),'--session-hash',str(info['session_hash']),
            '--tokenizer',info['tokenizer'],'--host','127.0.0.1','--port',str(a.port),
            '--context-limit',str(info['context_limit']),
            *(['--enable-mtp','--mtp-max-draft',str(a.mtp_max_draft)] if a.tp_batched_mtp else []),
            *(['--default-mode','mtp','--default-mtp-width','4'] if profile else [])])
        return sidecar.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if sidecar is not None: terminate(sidecar)
        if executor is not None: terminate(executor)
if __name__=='__main__':raise SystemExit(main())
