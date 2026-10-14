#!/usr/bin/env python3
"""Build an opt-in TP2 artifact view with replicated small control matrices.

Immutable segments are hard-linked, so this requires a common filesystem and
adds only indexes/manifest on disk. Source and existing TP artifacts are untouched.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from q38_shard_plan import load_stage


def replicate(name: str, profile: str = 'controls') -> bool:
    if profile == 'mtp':
        # Keep routed experts partitioned; shared MLP and all MTP controls/
        # QSA are full copies, matching the shared-down runtime contract.
        return (name == 'token_embd.weight' or name.startswith('mtp.')) and not name.endswith(
            ('.ffn_gate_up_exps.weight', '.ffn_down_exps.weight'))
    if name.startswith('mtp.'):
        return False
    if profile == 'attention-inputs' and name.endswith(
            ('.linear_attn.qkv.weight', '.linear_attn.z.weight', '.attn_q.weight')):
        return True
    return ('.hc_attn.' in name or '.hc_ffn.' in name or name.startswith('hc_input.')
            or name.endswith(('.ffn_gate_shexp.weight', '.ffn_up_shexp.weight',
                              '.linear_attn.in_a.weight', '.linear_attn.in_b.weight')))


def verify_extent(path: Path, offset: int, size: int, expected: str) -> None:
    h = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        remaining = size
        while remaining:
            buf = f.read(min(remaining, 8 << 20))
            if not buf:
                raise ValueError(f'short extent: {path}')
            h.update(buf)
            remaining -= len(buf)
    if h.hexdigest() != expected:
        raise ValueError(f'tensor hash mismatch: {path} offset {offset}')


def create(source: Path, tp: Path, output: Path, profile: str = 'controls') -> dict:
    if profile not in ('controls', 'attention-inputs', 'mtp'):
        raise ValueError(f'unknown replication profile: {profile}')
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    stages = [load_stage(source / f'stage{s}/index.q38d') for s in range(2)]
    ranks = [[load_stage(tp / f'rank{r}/stage{s}/index.q38d') for s in range(2)] for r in range(2)]
    for s, original in enumerate(stages):
        for rank in ranks:
            part = rank[s]
            if (part.source_commit, part.policy_sha256, part.cut, part.stage) != (
                    original.source_commit, original.policy_sha256, original.cut, original.stage):
                raise ValueError('source and TP artifact provenance differ')
    output.mkdir(parents=True)
    summary = {'schema': 'Q38_TP2_REPLICATED_CONTROLS_V1', 'source': str(source.resolve()),
               'tp_source': str(tp.resolve()), 'profile': profile, 'stages': []}
    verified = set()
    for r in range(2):
        for s in range(2):
            original, part = stages[s], ranks[r][s]
            original_by_name = {t.name: t for t in original.tensors}
            destination = output / f'rank{r}/stage{s}'
            destination.mkdir(parents=True)
            segments = []
            segment_map = {}
            tensors = []
            replaced = []
            added = 0
            part_by_name = {t.name: t for t in part.tensors}
            names = list(part_by_name)
            if profile == 'mtp' and s == 1:
                names.extend(t.name for t in original.tensors
                             if t.name not in part_by_name and replicate(t.name, profile))
            for tensor_name in names:
                tensor = part_by_name.get(tensor_name)
                full = original_by_name.get(tensor_name)
                selected = replicate(tensor_name, profile) and (profile != 'mtp' or s == 1)
                if selected and full is None:
                    raise ValueError(f'missing source tensor: {tensor_name}')
                replace = selected and (profile == 'mtp' or full.shape != tensor.shape)
                owner = original if replace else part
                chosen = full if replace else tensor
                owner_root = source / f'stage{s}' if replace else tp / f'rank{r}/stage{s}'
                key = (replace, chosen.segment)
                segment = owner.segments[chosen.segment]
                path = owner_root / segment.relative_path
                if replace:
                    verification = (s, chosen.name)
                    if verification not in verified:
                        verify_extent(path, chosen.data_offset, chosen.data_bytes, chosen.data_sha256)
                        if chosen.scale_bytes:
                            verify_extent(path, chosen.scale_offset, chosen.scale_bytes, chosen.scale_sha256)
                        verified.add(verification)
                    added += chosen.data_bytes + chosen.scale_bytes - (
                        tensor.data_bytes + tensor.scale_bytes if tensor else 0)
                    replaced.append(chosen.name)
                if key not in segment_map:
                    if path.stat().st_size != segment.bytes:
                        raise ValueError(f'segment extent mismatch: {path}')
                    name = f'{"source" if replace else "tp"}-{chosen.segment:04d}.q38w'
                    os.link(path, destination / name)
                    segment_map[key] = len(segments)
                    segments.append(dataclasses.replace(segment, relative_path=name))
                tensors.append(dataclasses.replace(chosen, segment=segment_map[key]))
            digest = hashlib.sha256(''.join(seg.sha256 for seg in segments).encode()).hexdigest()
            lines = ['Q38_DEVICE_INDEX_V1', f'stage={s}', f'cut={part.cut}',
                     f'source_repo={part.source_repo}', f'source_commit={part.source_commit}',
                     f'policy_sha256={part.policy_sha256}', f'artifact_sha256={digest}']
            lines += [f'segment\t{seg.relative_path}\t{seg.bytes}\t{seg.sha256}' for seg in segments]
            for t in tensors:
                lines.append('\t'.join(map(str, ['tensor', t.name, t.source_name, t.source_dtype,
                    t.fmt, t.group_size, t.segment, t.data_offset, t.data_bytes, t.data_sha256,
                    t.scale_offset, t.scale_bytes, t.scale_sha256 or '-', ','.join(map(str,t.shape))])))
            index = destination / 'index.q38d'
            index.write_text('\n'.join(lines)+'\n')
            load_stage(index)
            summary['stages'].append({'rank':r,'stage':s,'replicated':replaced,
                'added_resident_weight_bytes':added, 'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()})
    (output/'variant_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tp',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile', choices=('controls','attention-inputs','mtp'),default='controls')
    a=p.parse_args()
    result=create(a.source,a.tp,a.output,a.profile)
    print(json.dumps({'output':str(a.output),'added_weight_bytes_per_rank':[
        sum(s['added_resident_weight_bytes'] for s in result['stages'] if s['rank']==r)
        for r in range(2)]},indent=2))
if __name__=='__main__':main()
