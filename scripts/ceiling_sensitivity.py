from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import name_sweep_metrics as nsm
import contrast_metrics as cm
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'outputs' / 'metrics' / 'ceiling_sensitivity.csv'
CEILING_JSON = REPO / 'results' / 'expert_ceiling.json'
REFERENCE = REPO / 'results' / 'metrics' / 'convergence_aggregates.csv'
RHOS = ('0.0', '0.3', '0.5')

def main() -> None:
    traits = json.loads(CEILING_JSON.read_text())['traits']
    points = nsm.point_estimates(nsm.load_scored())
    crit = nsm.load_criterion()
    leaders = sorted(crit['leader_id'])
    expert = crit.set_index('leader_id').reindex(leaders)
    Xe = np.array([expert[f'expert_{t}_mean'].to_numpy(float) for t in nsm.config.TRAITS])
    rows = []
    for rho in RHOS:
        ceilings = {t: traits[t]['by_rho'][rho]['ceiling'] for t in traits}
        cvec = np.array([ceilings[t] for t in nsm.config.TRAITS])
        entries=[]
        for model in cm.MODEL_ORDER:
            Xa=cm._model_trait_matrix(points,leaders,model,'O',nsm.config.TRAITS)
            entries.append(dict(stratum='c1s',model=model,framing='O',trait_group='overall_mean',**cm._aggregate_level(Xe,Xa,cvec)))
        agg=pd.DataFrame(entries)
        rows.append(agg.assign(rho_w=float(rho), mean_ceiling=float(np.mean(list(ceilings.values())))))
    df = pd.concat(rows, ignore_index=True)
    ref = pd.read_csv(REFERENCE).query("stratum == 'c1s' and framing == 'O' and trait_group == 'overall_mean'").set_index('model')
    mid = df.query('rho_w == 0.3').set_index('model')
    cols = ['mean_r', 'ci_low', 'ci_high', 'mean_frac_ceiling', 'ci_low_frac', 'ci_high_frac']
    gap = (mid.loc[ref.index, cols] - ref[cols]).abs().to_numpy().max()
    print(f'rho_w = .3 vs convergence_aggregates.csv: max |diff| = {gap:.2e}')
    assert gap < 1e-12, "rho_w = .3 does not reproduce the paper's aggregate levels"
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT} ({len(df)} rows)')
    print(df.pivot(index='model', columns='rho_w', values='mean_frac_ceiling').round(3))
    print(df.drop_duplicates('rho_w')[['rho_w', 'mean_ceiling']].round(3).to_string(index=False))
if __name__ == '__main__':
    main()
