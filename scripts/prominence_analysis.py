"""Prominence tertile analyses using the bundled historical Wikimedia snapshot.

Current retrievals may differ from the historical measurement snapshot.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import fame_gradient_metrics as f
from prepare_records import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prominence', type=Path, default=f.PROMINENCE,
                        help='Historical snapshot by default; pass a fresh local CSV explicitly')
    args = parser.parse_args()
    prepare()
    f.OUT_DIR.mkdir(parents=True,exist_ok=True)
    resp=f.load_responses();pts=f.point_estimates(resp);abst=f.abstention_rates(resp)
    crit=pd.read_csv(f.CRITERION);prom=pd.read_csv(args.prominence).set_index('leader_id')
    if prom.index.duplicated().any():raise ValueError('Duplicate prominence politician identifiers')
    if set(prom.index)!=set(crit.leader_id):raise ValueError('Prominence measurements must cover the fixed sample')
    exp=crit.set_index('leader_id')[[f'expert_{t}_mean' for t in f.TRAITS]];exp.columns=f.TRAITS
    est={}
    for grp in f.MODEL_GROUPS:
        for fr in f.FRAMINGS:
            est[(grp,fr)]=pts[(pts.model_group==grp)&(pts.framing==fr)].pivot(index='leader_id',columns='trait',values='est').reindex(columns=f.TRAITS)
    ai=abst.set_index(['model_group','framing','leader_id']).sort_index()
    rows=[];gaps=[]
    for proxy,col in f.PROXIES.items():
        pv=prom[col].dropna();tert,q1,q2=f.assign_tertiles(pv)
        for grp in f.MODEL_GROUPS:
            for fr in f.FRAMINGS:
                ew=est[(grp,fr)]
                for level in ('low','mid','high'):
                    ids=[i for i in tert.index[tert==level] if i in ew.index]
                    if len(ids)<3:raise ValueError('Too few measurements for a prominence tertile')
                    llm_all,exp_all=f._mean_r(ids,ew,exp,f.TRAITS)
                    lo,hi=f._boot_ci(lambda idx:np.nanmean(f._per_trait_r(llm_all[idx],exp_all[idx])),len(ids))
                    ab=ai.loc[(grp,fr)].reindex(sorted(ids))
                    rows.append(dict(proxy=proxy,stratum='c1s',model_group=grp,framing=fr,tertile=level,n_leaders=len(ids),tertile_lo_bound=q1,tertile_hi_bound=q2,
                        bigfive_mean_r=np.nanmean(f._per_trait_r(*f._mean_r(ids,ew,exp,f.BIG_FIVE))),
                        darktriad_mean_r=np.nanmean(f._per_trait_r(*f._mean_r(ids,ew,exp,f.DARK_TRIAD))),
                        overall_mean_r=np.nanmean(f._per_trait_r(llm_all,exp_all)),overall_r_ci_low=lo,overall_r_ci_high=hi,
                        abstention_rate=ab.all2s_rate.mean(),near_parked_rate=ab.near_parked_rate.mean()))
                low=[i for i in tert.index[tert=='low'] if i in ew.index];high=[i for i in tert.index[tert=='high'] if i in ew.index]
                gaps.append(dict(proxy=proxy,stratum='c1s',model_group=grp,framing=fr,**f.tertile_gap_boot(low,high,ew,exp)))
    pd.DataFrame(rows).round(4).to_csv(f.OUT_DIR/'fame_gradient_by_tertile.csv',index=False)
    pd.DataFrame(gaps).round(4).to_csv(f.OUT_DIR/'fame_gradient_tertile_gap.csv',index=False)
    print(f'Wrote {len(rows)} tertile and {len(gaps)} gap rows from local prominence measurements.')

if __name__=='__main__':main()
