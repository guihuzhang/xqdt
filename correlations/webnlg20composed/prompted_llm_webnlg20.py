#!/usr/bin/env python3
"""Run prompted LLM error detection on WebNLG 2020 human-evaluation pairs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
COMMON=HERE.parent
sys.path.insert(0,str(COMMON))
from prompted_llm_common import add_backend_args, append_jsonl, build_prompt, load_jsonl, make_backend, model_slug, parse_response
from corr_webnlg20_composed import SYSTEM_NAME_MAP, build_alignment

DEFAULT_HUMAN=HERE/'human-annotations/human_eval_webnlg_20.json'
DEFAULT_ANCHOR_DIR=HERE/'xqdt_results'
DEFAULT_OUTPUT=HERE/'prompted_llm_results'


def compute_prf(errors,n):
 m,e,i=map(lambda k:len(errors[k]),('missing','extra','incorrect')); tp=max(0,n-m-i); fp=e+i; fn=m+i
 p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.; f=2*p*r/(p+r) if p+r else 0.
 return dict(precision=p,recall=r,f1=f,TP=tp,FP=fp,FN=fn)


def main():
 ap=argparse.ArgumentParser(description=__doc__); add_backend_args(ap)
 ap.add_argument('--human-json',type=Path,default=DEFAULT_HUMAN)
 ap.add_argument('--anchor-results-dir',type=Path,default=DEFAULT_ANCHOR_DIR)
 ap.add_argument('--anchor-eval',default='qwen3_8b_7479.json')
 ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
 ap.add_argument('--prepare-only',action='store_true'); args=ap.parse_args()
 aligned,_,name_map=build_alignment(args.human_json,args.anchor_results_dir,args.anchor_eval)
 data=[]
 for obj in aligned:
  for key,text in obj.items():
   if not key.endswith('-pred'): continue
   human_team=key[:-5]; team=name_map.get(human_team,SYSTEM_NAME_MAP.get(human_team,human_team))
   data.append(dict(sample_id=f"{team}_{obj['entry_id']}",team=team,entry_id=obj['entry_id'],text=text,triples=obj['table']))
 if args.max_samples: data=data[:args.max_samples]
 print(f'Prepared {len(data)} WebNLG 2020 human-evaluation pairs')
 if args.prepare_only: return
 args.output_dir.mkdir(parents=True,exist_ok=True); slug=model_slug(args)
 ckpt=args.output_dir/f'{slug}_results.jsonl'; final=args.output_dir/f'{slug}_results.json'
 if args.overwrite: ckpt.unlink(missing_ok=True); final.unlink(missing_ok=True)
 existing=load_jsonl(ckpt); pending=[x for x in data if x['sample_id'] not in existing]
 backend=make_backend(args)
 try:
  outputs=backend.infer([build_prompt(x['text'],x['triples']) for x in pending])
  for item,out in tqdm(zip(pending,outputs),total=len(pending),desc='Saving',unit='pair'):
   errors=parse_response(out); rec={**item,'model':args.model,'model_output':out,'parsed_errors':errors,**compute_prf(errors,len(item['triples']))}
   append_jsonl(ckpt,rec); existing[item['sample_id']]=rec
 finally: backend.close()
 ordered=[existing[x['sample_id']] for x in data]; final.write_text(json.dumps(ordered,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Saved {len(ordered)} pairs to {final}')
 print('Next: build_metric_results_from_xqdt.py --results-dir prompted_llm_results --selected '+slug)

if __name__=='__main__': main()
