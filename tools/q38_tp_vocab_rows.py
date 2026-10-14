#!/usr/bin/env python3
"""Immutable lossless vocabulary-row TP variant for the native LM head."""
from pathlib import Path
import argparse,dataclasses,hashlib,json,os
from q38_shard_plan import load_stage,Segment

def digest(data):return hashlib.sha256(data).hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def create(ready,tp,output):
 if output.exists():raise ValueError('output already exists')
 metadata=json.loads(ready.read_text());source=load_stage(Path(metadata['stage1_index']))
 head=next(t for t in source.tensors if t.name=='output.weight')
 require(head.shape==(248320,2560) and head.group_size==128 and head.fmt=='w8a16_sym_g128','expected W8 G128 full vocabulary head')
 segment=Path(metadata['stage1_index']).parent/source.segments[head.segment].relative_path
 with segment.open('rb') as f:
  f.seek(head.data_offset);data=f.read(head.data_bytes);f.seek(head.scale_offset);scales=f.read(head.scale_bytes)
 require(digest(data)==head.data_sha256 and digest(scales)==head.scale_sha256,'source vocabulary head checksum mismatch')
 manifest=json.loads((tp/'variant_manifest.json').read_text());manifest['parent']=str(tp);manifest['profile']='vocabulary-rows';manifest['vocabulary_head_layout']='rows-v1';manifest['head_source_quant_sha256']=head.data_sha256;manifest['head_source_scales_sha256']=head.scale_sha256
 require({(rec['rank'],rec['stage']) for rec in manifest['stages']}=={(r,s) for r in (0,1) for s in (0,1)} and len(manifest['stages'])==4,'expected exactly four TP2 stage records')
 output.mkdir(parents=True)
 for rec in manifest['stages']:
  r,s=rec['rank'],rec['stage'];old=tp/f'rank{r}/stage{s}';dest=output/f'rank{r}/stage{s}';dest.mkdir(parents=True)
  layout=load_stage(old/'index.q38d')
  for seg in layout.segments:os.link(old/seg.relative_path,dest/seg.relative_path)
  if s==1:
   q=data[r*(len(data)//2):(r+1)*(len(data)//2)];sc=scales[r*(len(scales)//2):(r+1)*(len(scales)//2)]
   packed=q+sc;name='vocabulary-rows.q38w';(dest/name).write_bytes(packed)
   new=dataclasses.replace(head,shape=(124160,2560),segment=len(layout.segments),data_offset=0,data_bytes=len(q),data_sha256=digest(q),scale_offset=len(q),scale_bytes=len(sc),scale_sha256=digest(sc))
   layout.segments.append(Segment(name,len(packed),digest(packed)))
   layout.tensors=[new if t.name=='output.weight' else t for t in layout.tensors]
   rec['head_rows']=[r*124160,(r+1)*124160];rec['head_quant_sha256']=digest(q);rec['head_scales_sha256']=digest(sc)
  lines=['Q38_DEVICE_INDEX_V1',f'stage={s}',f'cut={layout.cut}',f'source_repo={layout.source_repo}',f'source_commit={layout.source_commit}',f'policy_sha256={layout.policy_sha256}',f'artifact_sha256={digest("".join(seg.sha256 for seg in layout.segments).encode())}']
  lines += [f'segment\t{seg.relative_path}\t{seg.bytes}\t{seg.sha256}' for seg in layout.segments]
  for t in layout.tensors:lines.append('\t'.join(map(str,['tensor',t.name,t.source_name,t.source_dtype,t.fmt,t.group_size,t.segment,t.data_offset,t.data_bytes,t.data_sha256,t.scale_offset,t.scale_bytes,t.scale_sha256 or '-',','.join(map(str,t.shape))])))
  index=dest/'index.q38d';index.write_text('\n'.join(lines)+'\n');load_stage(index);rec['index_sha256']=digest(index.read_bytes())
 (output/'variant_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return {'output':str(output),'quant_rows_reconstruct_source':digest(b''.join((output/f'rank{r}/stage1/vocabulary-rows.q38w').read_bytes()[:head.data_bytes//2] for r in range(2)))==head.data_sha256,'scale_rows_reconstruct_source':digest(b''.join((output/f'rank{r}/stage1/vocabulary-rows.q38w').read_bytes()[head.data_bytes//2:] for r in range(2)))==head.scale_sha256}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--tp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(create(a.ready,a.tp,a.output),indent=2))
