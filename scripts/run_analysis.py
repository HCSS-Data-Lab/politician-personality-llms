"""Recompute the primary statistics from frozen model responses and local NEGex.

No inference calls are made. External data are never bundled in the release.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_records import prepare
import name_sweep_metrics as nsm
import contrast_metrics as cm
import speech_metrics as sm
import speech_model_contrasts as sc


def save(frame,name):
    path=ROOT/'outputs/metrics'/f'{name}.csv'
    frame.to_csv(path,index=False)
    print(f'{name}: {len(frame)} rows')


def recognition(records_dir):
    rows=[json.loads(l) for l in (records_dir/'recognition.jsonl').open()]
    parts=[]
    for model in sorted({r['model'] for r in rows})+['open_panel']:
        selected=[r for r in rows if (r['model']!='gpt-5.5' if model=='open_panel' else r['model']==model)]
        n=len(selected);correct=sum(r['label']=='correct' for r in selected)
        partial=sum(r['label']=='partial' for r in selected)
        parts.append(dict(model=model,n=n,correct=correct,partial=partial,
                          strict=correct/n,lenient=(correct+partial)/n))
    save(pd.DataFrame(parts),'recognition')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['all','name','speech','recognition'],default='all')
    ap.add_argument('--records-dir',type=Path,default=ROOT/'records',help='Frozen exports or a complete fresh run directory; originals are never overwritten.')
    args=ap.parse_args()
    prepare(records_dir=args.records_dir)
    recognition(args.records_dir)
    if args.stage=='recognition':return
    if not nsm.CRITERION.exists():
        raise SystemExit('Download NEGex 2.0 and run scripts/build_inputs.py first; see docs/data-access.md.')
    # Ceilings are aggregate reference estimates, recomputable via expert_ceiling.py.
    nsm.CEILING_JSON=ROOT/'results/expert_ceiling.json'
    scored=nsm.load_scored();crit=nsm.load_criterion();ceil=nsm.load_ceilings()
    points=nsm.point_estimates(scored)
    if args.stage in ('all','name'):
        save(nsm.convergence_table(points,crit,ceil,'c1s'),'convergence')
        save(nsm.bias_table(points,crit,'c1s'),'directional_bias')
        rel=nsm.reliability_table(scored,crit,'c1s')
        save(rel,'reliability_icc');save(nsm.reliability_rollup(rel),'reliability_icc_rollup')
        save(cm.build_framing_contrasts(points,crit),'framing_contrasts')
        save(cm.build_convergence_aggregates(points,crit,ceil),'convergence_aggregates')
        save(cm.build_model_contrasts(points,crit,ceil),'model_contrasts')
    if args.stage in ('all','speech'):
        criterion={r['leader_id']:r for r in crit.to_dict('records')}
        npnt=sm.name_points();spnt,_=sm.speech_points()
        recog,_=sm.load_recognition()
        save(sm.build_speech_convergence(npnt,spnt,criterion,ceil),'speech_convergence')
        save(sm.build_evidence_contrasts(npnt,spnt,criterion),'evidence_contrasts')
        save(sm.build_anon_by_recognition(sm.anon_speech_rows(),recog,criterion,ceil),'anon_by_recognition')
        save(sm.build_speech_reliability(sm.speech_scored()),'speech_reliability_icc')
        save(sm.build_speech_bias(spnt,criterion),'speech_directional_bias')
        save(sm.build_speech_framing_contrasts(spnt,criterion),'speech_framing_contrasts')
        save(sc.build_speech_model_contrasts(npnt,spnt,criterion,ceil),'speech_model_contrasts')

if __name__=='__main__':main()
