#!/usr/bin/env python3
"""Score saved prompted-LLM E2E outputs with the paper's correlation protocol.

Applies the drop_gt100_all+A+C+D+E+F worker-filtering strategy (via
rebuild_human_quality_scores) before correlating, matching the published
Table 19 methodology. Without this step the output does not match the paper.
"""
import argparse, json
from pathlib import Path
from e2e_correlation_light import compute_correlations, rebuild_human_quality_scores

DEFAULT_CONVERTED = Path(__file__).resolve().parent / 'human_ratings/converted.json'

def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--model-name',default=None); ap.add_argument('--n-bootstrap',type=int,default=1000); ap.add_argument('--output',type=Path,default=None); ap.add_argument('--converted',type=Path,default=DEFAULT_CONVERTED); args=ap.parse_args()
 rows=json.loads(args.results.read_text(encoding='utf-8')); name=args.model_name or args.results.stem.removesuffix('_results')
 converted=json.loads(args.converted.read_text(encoding='utf-8')); rows=rebuild_human_quality_scores(rows,converted)
 result=compute_correlations(rows,name,n_bootstrap=args.n_bootstrap); out=args.output or args.results.with_name(args.results.stem.removesuffix('_results')+'_correlation.json')
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'Saved: {out}')
if __name__=='__main__':main()
