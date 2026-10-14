#!/usr/bin/env python3
"""Losslessly pack owned QKV/Z rows; retain full B/A and bulk projections."""
from pathlib import Path
import argparse,dataclasses,hashlib,json,os
from q38_shard_plan import load_stage,Segment,Tensor

def digest(data):return hashlib.sha256(data).hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def create(ready,tp,output):
 if output.exists():raise ValueError('output already exists')
 metadata=json.loads(ready.read_text());sources={};evidence=[]
 for stage in (0,1):
  index=Path(metadata[f'stage{stage}_index']);layout=load_stage(index)
  for t in layout.tensors:
   if any(t.name.endswith('.linear_attn.'+name+'.weight') for name in ('qkv','z','in_b','in_a')):
    require(t.shape[1:]==(2560,) and t.shape[0] in (10240,6144,48) and t.fmt=='w8a16_sym_g128' and t.group_size==128,'expected W8 G128 GDN projection: '+t.name)
    with (index.parent/layout.segments[t.segment].relative_path).open('rb') as f:
     f.seek(t.data_offset);q=f.read(t.data_bytes);f.seek(t.scale_offset);sc=f.read(t.scale_bytes)
    require(digest(q)==t.data_sha256 and digest(sc)==t.scale_sha256,'source GDN checksum mismatch: '+t.name)
    sources[t.name]=(t,q,sc)
 manifest=json.loads((tp/'variant_manifest.json').read_text());manifest['parent']=str(tp);manifest['profile']='vocabulary-rows-gdn-heads';manifest['gdn_qkv_layout']='packed-qkvz-8192-bulk-original';
 require({(rec['rank'],rec['stage']) for rec in manifest['stages']}=={(r,s) for r in (0,1) for s in (0,1)} and len(manifest['stages'])==4,'expected exactly four TP2 stage records')
 output.mkdir(parents=True)
 for rec in manifest['stages']:
  r,s=rec['rank'],rec['stage'];old=tp/f'rank{r}/stage{s}';dest=output/f'rank{r}/stage{s}';dest.mkdir(parents=True);layout=load_stage(old/'index.q38d')
  for seg in layout.segments:os.link(old/seg.relative_path,dest/seg.relative_path)
  packed=bytearray(); additions=[]
  for name,(t,q,sc) in sources.items():
   if not name.endswith('.qkv.weight') or not any(old_t.name==name for old_t in layout.tensors):continue
   root=name.removesuffix('qkv.weight'); qb=bytearray();sb=bytearray(); proof=[]
   for sub in ('qkv','z'):
    st,sq,ss=sources[root+sub+'.weight']
    spans=[(r*1024,(r+1)*1024),(2048+r*1024,2048+(r+1)*1024),(4096+r*3072,4096+(r+1)*3072)] if sub=='qkv' else [(r*st.shape[0]//2,(r+1)*st.shape[0]//2)]
    for a,b in spans:qb+=sq[a*2560:b*2560];sb+=ss[a*40:b*40]
    proof.append(dict(tensor=st.name,rows=spans,data_sha256=st.data_sha256,scale_sha256=st.scale_sha256))
   require(len(qb)==8192*2560 and len(sb)==8192*40,'invalid packed QKV/Z extent')
   offset=len(packed);packed+=qb;scaleoff=len(packed);packed+=sb
   additions.append(dataclasses.replace(t,name=root+'qkvzba_packed.weight',source_name=root+'qkvzba_packed.weight',shape=(8192,2560),segment=len(layout.segments),data_offset=offset,data_bytes=len(qb),data_sha256=digest(qb),scale_offset=scaleoff,scale_bytes=len(sb),scale_sha256=digest(sb)))
   evidence.append(dict(rank=r,stage=s,tensor=additions[-1].name,sources=proof,data_sha256=digest(qb),scale_sha256=digest(sb)))
  layout.tensors+=additions
  marker=b'\0\0\0\0';markname=f'q38.tp.gdn_heads.stage{s}.rank{r}';layout.tensors.append(Tensor(markname,markname,'BF16','preserve',0,len(layout.segments),len(packed),4,digest(marker),0,0,'',(2,)));packed+=marker
  layout.tensors.append(Tensor(f'q38.tp.gdn_packed.stage{s}.rank{r}',markname,'BF16','preserve',0,len(layout.segments),len(packed),4,digest(marker),0,0,'',(2,)));packed+=marker
  segname='gdn-packed.q38w';(dest/segname).write_bytes(packed);layout.segments.append(Segment(segname,len(packed),digest(packed)))
  lines=['Q38_DEVICE_INDEX_V1',f'stage={s}',f'cut={layout.cut}',f'source_repo={layout.source_repo}',f'source_commit={layout.source_commit}',f'policy_sha256={layout.policy_sha256}',f'artifact_sha256={digest("".join(seg.sha256 for seg in layout.segments).encode())}']
  lines += [f'segment\t{seg.relative_path}\t{seg.bytes}\t{seg.sha256}' for seg in layout.segments]
  for t in layout.tensors:lines.append('\t'.join(map(str,['tensor',t.name,t.source_name,t.source_dtype,t.fmt,t.group_size,t.segment,t.data_offset,t.data_bytes,t.data_sha256,t.scale_offset,t.scale_bytes,t.scale_sha256 or '-',','.join(map(str,t.shape))])))
  index=dest/'index.q38d';index.write_text('\n'.join(lines)+'\n');load_stage(index);rec['index_sha256']=digest(index.read_bytes());rec['gdn_heads']=[r*24,(r+1)*24]
 manifest['gdn_source_evidence']=evidence;(output/'variant_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return dict(output=str(output),tensors=len(evidence),lossless_source_hashes_verified=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--tp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(create(a.ready,a.tp,a.output),indent=2))
