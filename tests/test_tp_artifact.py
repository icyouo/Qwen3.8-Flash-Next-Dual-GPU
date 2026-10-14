from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from q38_shard_plan import load_stage
from q38_tp_replicated_controls import create, replicate

class ReplicatedControlsTest(unittest.TestCase):
    def write_stage(self, root, stage, full, bad_hash=False):
        root.mkdir(parents=True)
        data=bytes(range(16 if full else 8))
        path=root/'data.bin'; path.write_bytes(data)
        digest=hashlib.sha256(data).hexdigest()
        tensor_digest='0'*64 if bad_hash else digest
        rows=4 if full else 2
        (root/'index.q38d').write_text('\n'.join([
            'Q38_DEVICE_INDEX_V1',f'stage={stage}','cut=25','source_repo=Qwen/Qwen3.8-Flash-Next',
            'source_commit='+'a'*40,'policy_sha256='+'b'*64,'artifact_sha256='+digest,
            f'segment\tdata.bin\t{len(data)}\t{digest}',
            f'tensor\tblk.{stage}.hc_attn.mix_down.weight\tsource.weight\tBF16\tpreserve\t0\t0\t0\t{len(data)}\t{tensor_digest}\t0\t0\t-\t{rows},2',
        ])+'\n')
    def fixture(self, root, bad_hash=False):
        source=root/'source';tp=root/'tp'
        for stage in range(2):
            self.write_stage(source/f'stage{stage}',stage,True,bad_hash)
            for rank in range(2):self.write_stage(tp/f'rank{rank}/stage{stage}',stage,False)
        return source,tp
    def test_replicates_hashed_extents_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,tp=self.fixture(root)
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            manifest=create(source,tp,root/'out')
            self.assertEqual(len(manifest['stages']),4)
            for rank in range(2):
                for stage in range(2):
                    folder=root/f'out/rank{rank}/stage{stage}'
                    layout=load_stage(folder/'index.q38d')
                    self.assertEqual(layout.tensors[0].shape,(4,2))
                    linked=folder/layout.segments[0].relative_path
                    self.assertEqual(linked.stat().st_ino,(source/f'stage{stage}/data.bin').stat().st_ino)
            for path,data in before.items():self.assertEqual(path.read_bytes(),data)
            with self.assertRaisesRegex(ValueError,'exists'):create(source,tp,root/'out')
    def test_rejects_corrupt_source_tensor_before_manifest_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,tp=self.fixture(root,True)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):create(source,tp,root/'out')
            self.assertFalse((root/'out/variant_manifest.json').exists())
    def test_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,tp=self.fixture(root)
            index=tp/'rank1/stage1/index.q38d'
            index.write_text(index.read_text().replace('source_commit='+'a'*40,'source_commit='+'c'*40))
            with self.assertRaisesRegex(ValueError,'provenance'):create(source,tp,root/'out')
            self.assertFalse((root/'out').exists())
    def test_mtp_adds_missing_rank_tensors_and_preserves_expert_shards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,tp=self.fixture(root)
            names=['mtp.fc_hidden.weight','mtp.blk.0.attn_output.weight',
                   'mtp.blk.0.ffn_gate_up_exps.weight','mtp.blk.0.ffn_down_shexp.weight']
            for folder in [source/'stage1',tp/'rank0/stage1',tp/'rank1/stage1']:
                index=folder/'index.q38d'
                lines=index.read_text().splitlines()
                template=lines.pop().split('\t')
                selected=names[2:] if folder==tp/'rank0/stage1' else names
                for name in selected:
                    row=template.copy();row[1]=name;lines.append('\t'.join(row))
                index.write_text('\n'.join(lines)+'\n')
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            create(source,tp,root/'out','mtp')
            for rank in range(2):
                folder=root/f'out/rank{rank}/stage1'
                stage=load_stage(folder/'index.q38d')
                tensors={t.name:t for t in stage.tensors}
                self.assertEqual(set(tensors),set(names))
                for name in names:
                    full=name != names[2]
                    t=tensors[name]
                    self.assertEqual(t.shape,(4 if full else 2,2))
                    owner=source/'stage1' if full else tp/f'rank{rank}/stage1'
                    self.assertEqual((folder/stage.segments[t.segment].relative_path).stat().st_ino,
                                     (owner/'data.bin').stat().st_ino)
                self.assertEqual(load_stage(root/f'out/rank{rank}/stage0/index.q38d').tensors[0].shape,(2,2))
            for path,data in before.items():self.assertEqual(path.read_bytes(),data)
            self.assertTrue(replicate('token_embd.weight','mtp'))
            self.assertFalse(replicate('output.weight','mtp'))
    def test_unknown_profile_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,tp=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'unknown replication profile'):
                create(source,tp,root/'out','typo')
            self.assertFalse((root/'out').exists())
if __name__=='__main__':unittest.main()
