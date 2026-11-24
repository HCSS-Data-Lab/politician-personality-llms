from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import name_sweep_metrics as nsm
from fame_gradient_metrics import SEED, N_BOOT, PROXIES, assign_tertiles
REPO = Path(__file__).resolve().parents[1]
PROMINENCE = REPO / 'measurements' / 'prominence_2026-07-06.csv'
OUT = REPO / 'outputs' / 'fame_gradient' / 'desirability_by_prominence.csv'
DESIRABLE = ['agreeableness', 'conscientiousness', 'emotional_stability']
TRAITS = DESIRABLE + ['psychopathy']
MODELS = ['gpt-5.5', 'pooled', 'gemma-27b', 'gemma-12b', 'qwen-32b', 'qwen-8b']
METRICS = ('desirable', 'psychopathy_under', 'combined')
REPORTED = {('pageviews', 'gpt-5.5', 'desirable'): (+0.013, -0.227, +0.255), ('pageviews', 'gpt-5.5', 'psychopathy_under'): (-0.346, -0.746, +0.078), ('pageviews', 'gpt-5.5', 'combined'): (-0.06, -0.308, +0.187), ('pageviews', 'pooled', 'desirable'): (-0.037, -0.28, +0.226), ('pageviews', 'pooled', 'psychopathy_under'): (-0.113, -0.61, +0.384), ('pageviews', 'pooled', 'combined'): (-0.087, -0.357, +0.167), ('sitelinks', 'gpt-5.5', 'desirable'): (-0.113, -0.365, +0.146), ('sitelinks', 'gpt-5.5', 'psychopathy_under'): (-0.495, -0.878, -0.091), ('sitelinks', 'gpt-5.5', 'combined'): (-0.226, -0.472, +0.039), ('sitelinks', 'pooled', 'desirable'): (-0.178, -0.433, +0.088), ('sitelinks', 'pooled', 'psychopathy_under'): (-0.206, -0.631, +0.2), ('sitelinks', 'pooled', 'combined'): (-0.178, -0.424, +0.091), ('pageviews', 'gemma-12b', 'combined'): (-0.452, -0.771, -0.16), ('sitelinks', 'gemma-12b', 'combined'): (-0.463, -0.765, -0.122)}

def leader_scores() -> pd.DataFrame:
    pts = nsm.point_estimates(nsm.load_scored())
    crit = nsm.load_criterion().query("stratum == 'c1s'")
    parts = []
    for t in TRAITS:
        x = pts.query("framing == 'O' and model in @MODELS and trait == @t").merge(crit[['leader_id', f'expert_{t}_mean', f'expert_{t}_sd']], on='leader_id')
        x['gap'] = (x.llm_mean - x[f'expert_{t}_mean']) / x[f'expert_{t}_sd'].replace(0, np.nan)
        parts.append(x[['leader_id', 'model', 'gap']].assign(trait=t))
    w = pd.concat(parts).pivot_table(index=['leader_id', 'model'], columns='trait', values='gap').reset_index()
    w['desirable'] = w[DESIRABLE].mean(axis=1)
    w['psychopathy_under'] = -w['psychopathy']
    w['combined'] = pd.concat([w[t] for t in DESIRABLE] + [-w['psychopathy']], axis=1).mean(axis=1)
    return w

def main() -> None:
    parser = argparse.ArgumentParser(description='Desirability by prominence using the historical Wikimedia snapshot')
    parser.add_argument('--prominence', type=Path, default=PROMINENCE,
                        help='Historical snapshot by default; pass a fresh local CSV explicitly')
    args = parser.parse_args()
    w = leader_scores()
    prom = pd.read_csv(args.prominence).query("source_stratum == 'c1s'")
    expected = set(nsm.load_criterion().leader_id)
    if prom.leader_id.duplicated().any() or set(prom.leader_id) != expected:
        raise ValueError('Prominence measurements must cover the fixed sample exactly')
    rows = []
    for proxy, col in PROXIES.items():
        p = prom[['leader_id', col]].dropna()
        p = p.assign(tertile=assign_tertiles(p[col])[0])
        z = w.merge(p, on='leader_id')
        for m in MODELS:
            q = z[z.model == m]
            for metric in METRICS:
                a = {k: q.loc[q.tertile == k, metric].dropna().to_numpy(float) for k in ('low', 'mid', 'high')}
                hi, lo = (a['high'], a['low'])
                rng = np.random.default_rng(SEED)
                boots = np.array([np.mean(hi[rng.integers(0, len(hi), len(hi))]) - np.mean(lo[rng.integers(0, len(lo), len(lo))]) for _ in range(N_BOOT)])
                lo_ci, hi_ci = np.quantile(boots, [0.025, 0.975])
                rows.append({'proxy': proxy, 'model': m, 'metric': metric, 'framing': 'O', 'n_low': len(lo), 'n_mid': len(a['mid']), 'n_high': len(hi), 'mean_low': lo.mean(), 'mean_mid': a['mid'].mean(), 'mean_high': hi.mean(), 'high_minus_low': hi.mean() - lo.mean(), 'ci_low': lo_ci, 'ci_high': hi_ci, 'p_two_sided': min(1.0, 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0))))})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT} ({len(df)} rows)')
    verify_historical_results(df, args.prominence)


def verify_historical_results(df: pd.DataFrame, prominence_path: Path) -> None:
    if Path(prominence_path).resolve() != PROMINENCE.resolve():
        print('Alternative prominence measurements selected; historical comparison not applied.')
        return
    idx = df.set_index(['proxy', 'model', 'metric'])
    bad = 0
    for key, want in REPORTED.items():
        got = tuple((round(float(idx.loc[key, c]), 3) for c in ('high_minus_low', 'ci_low', 'ci_high')))
        ok = all((abs(g - w_) < 0.0005 for g, w_ in zip(got, want)))
        bad += not ok
        print(f"{('OK ' if ok else 'BAD')} {key}: got {got} reported {want}")
    print(f'reproduction: {len(REPORTED) - bad}/{len(REPORTED)} reported numbers match')
    if bad:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
