#!/usr/bin/env python3
"""Run prompted LLM error detection on E2E human-evaluation pairs."""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

HERE=Path(__file__).resolve().parent; COMMON=HERE.parent; sys.path.insert(0,str(COMMON))
from prompted_llm_common import add_backend_args, append_jsonl, build_prompt, load_jsonl, make_backend, model_slug, parse_response
DEFAULT_INPUT=HERE/'human_ratings/converted.json'; DEFAULT_OUTPUT=HERE/'prompted_llm_results'


def norm_slot(s):
 s=s.strip().replace('_',' '); s=re.sub(r'([a-z])([A-Z])',r'\1 \2',s); return re.sub(r'\s+',' ',s).strip().lower()
def parse_mr(text):
 pairs=[(m.group(1).strip(),m.group(2).strip()) for m in re.finditer(r'\s*([^,\[]+?)\s*\[([^\]]*)\]\s*(?:,|$)',text)]
 subject=next((v for s,v in pairs if norm_slot(s)=='name' and v),'restaurant')
 return [f'[S] {subject} [P] {norm_slot(s)} [O] {v}' for s,v in pairs if v and norm_slot(s)!='name']
def flatten(data):
 out=[]
 for mr_id,mr in data.items():
  for sys_name,sd in mr['systems'].items():
   quality=sd.get('quality',[]); labels=[re.sub(r'\s+',' ',str(a.get('coarse_label','')).strip().lower()) for a in quality if str(a.get('coarse_label','')).strip()]
   counts=dict(Counter(labels)); majority=sorted(counts.items(),key=lambda x:(-x[1],x[0]))[0][0] if counts else None
   out.append(dict(sample_id=f'{mr_id}_{sys_name}',id=f'{mr_id}_{sys_name}',mr_id=str(mr_id),sys_name=sys_name,mr=mr['mr'],output=sd['output'],triples=parse_mr(mr['mr']),human_quality_scores=[a['fine_score'] for a in quality],human_quality_coarse_labels=labels,human_quality_coarse_counts=counts,human_quality_is_ok=[x=='ok' for x in labels],human_quality_ok_ratio=(sum(x=='ok' for x in labels)/len(labels) if labels else None),human_quality_majority_coarse=majority,human_quality_majority_is_ok=(majority=='ok' if majority else None),human_naturalness_scores=[a['fine_score'] for a in sd.get('naturalness',[])]))
 return out
def prf(err,n):
 m,e,i=map(lambda k:len(err[k]),('missing','extra','incorrect')); tp=max(0,n-m-i); fp=e+i; fn=m+i
 p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.; f=2*p*r/(p+r) if p+r else 0.
 return dict(precision=p,recall=r,f1=f,TP=tp,FP=fp,FN=fn)

def main():
 ap=argparse.ArgumentParser(description=__doc__); add_backend_args(ap); ap.add_argument('--input',type=Path,default=DEFAULT_INPUT); ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT); ap.add_argument('--prepare-only',action='store_true'); args=ap.parse_args()
 data=flatten(json.loads(args.input.read_text(encoding='utf-8')))
 if args.max_samples:data=data[:args.max_samples]
 print(f'Prepared {len(data)} E2E human-evaluation pairs')
 if args.prepare_only:return
 args.output_dir.mkdir(parents=True,exist_ok=True); slug=model_slug(args); ckpt=args.output_dir/f'{slug}_results.jsonl'; final=args.output_dir/f'{slug}_results.json'
 if args.overwrite:ckpt.unlink(missing_ok=True);final.unlink(missing_ok=True)
 existing=load_jsonl(ckpt); pending=[x for x in data if x['sample_id'] not in existing]; backend=make_backend(args)
 try:
  outputs=backend.infer([build_prompt(x['output'],x['triples']) for x in pending])
  for item,out in tqdm(zip(pending,outputs),total=len(pending),desc='Saving',unit='pair'):
   errors=parse_response(out); rec={**item,'model':args.model,'num_triples':len(item['triples']),'model_response':out,'parsed_errors':errors,'error_counts':{k:len(v) for k,v in errors.items()},**prf(errors,len(item['triples']))}
   rec.pop('sample_id'); append_jsonl(ckpt,{**rec,'sample_id':item['sample_id']}); existing[item['sample_id']]={**rec,'sample_id':item['sample_id']}
 finally:backend.close()
 ordered=[]
 for x in data:
  rec=dict(existing[x['sample_id']]); rec.pop('sample_id',None); ordered.append(rec)
 final.write_text(json.dumps(ordered,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'Saved {len(ordered)} pairs to {final}')
 print('Next: score_prompted_llm_e2e.py --results '+str(final))
if __name__=='__main__':main()
