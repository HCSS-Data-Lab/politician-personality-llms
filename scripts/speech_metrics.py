from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
for p in (str(REPO), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import name_sweep_metrics as nsm
from pinf import config, metrics
from pinf.anonymize import score_reid_guess
SEED = nsm.SEED
N_BOOT = nsm.N_BOOT
CI_LEVEL = nsm.CI_LEVEL
OPEN_MODELS = nsm.OPEN_MODELS
POOLED = nsm.POOLED
TRAITS = config.TRAITS
BIG_FIVE = TRAITS[:5]
DARK_TRIAD = TRAITS[5:]
GROUPS = {'big_five_mean': BIG_FIVE, 'dark_triad_mean': DARK_TRIAD, 'overall_mean': TRAITS}
SWEEP = REPO
SPEECH_FILE = SWEEP / 'outputs' / 'results_speech_20260706T131201.jsonl'
NAME_BRIDGE_FILE = SWEEP / 'outputs' / 'name_arm_bridge.jsonl'
REID_FILE = SWEEP / 'outputs' / 'reid_speech_20260706T131201.jsonl'
ANCHOR = 'gpt-5.5'
GPT_SPEECH_FILE = SWEEP / 'outputs' / 'results_speech_openrouter_gpt-5.5_20260716T131331.jsonl'
GPT_REID_FILE = SWEEP / 'outputs' / 'reid_speech_openrouter_gpt-5.5_20260716T131331.jsonl'
GPT_NAME_BRIDGE_FILE = SWEEP / 'outputs' / 'name_arm_bridge_gpt-5.5.jsonl'
SPEECH_FILES = [SPEECH_FILE, GPT_SPEECH_FILE]
REID_FILES = [REID_FILE, GPT_REID_FILE]
NAME_BRIDGE_FILES = [NAME_BRIDGE_FILE, GPT_NAME_BRIDGE_FILE]
ANCHORS = [ANCHOR]
CRITERION = SWEEP / 'data' / 'criterion.csv'
PROMINENCE = REPO / 'data' / 'prominence.csv'
PROMINENCE_FALLBACK = REPO / 'data' / 'prominence.csv'
OUT_DIR = SWEEP / 'outputs' / 'metrics'
LO, HI = ((1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100)

def _jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def name_points() -> pd.DataFrame:
    rows = []
    for path in NAME_BRIDGE_FILES:
        for r in _jsonl(path):
            if not r.get('parse_ok'):
                continue
            for trait, s in nsm.score_ratings(r['ratings']).items():
                rows.append((r['leader_id'], r['model'], r['framing'], trait, s))
    df = pd.DataFrame(rows, columns=['leader_id', 'model', 'framing', 'trait', 'score'])
    est = df.groupby(['leader_id', 'model', 'framing', 'trait'], as_index=False)['score'].mean().rename(columns={'score': 'value'})
    return _add_pooled(est, ['leader_id', 'framing', 'trait'])

def speech_points() -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    tok = defaultdict(list)
    for path in SPEECH_FILES:
        for r in _jsonl(path):
            if not r.get('parse_ok') or not r.get('name_arm_id'):
                continue
            if r['condition'] == 'named-speech' and r['model'] in OPEN_MODELS:
                tok[r['name_arm_id']].append(r.get('prompt_tokens', np.nan))
            for trait, s in nsm.score_ratings(r['ratings']).items():
                rows.append((r['name_arm_id'], r['condition'], r['model'], r['framing'], trait, s))
    df = pd.DataFrame(rows, columns=['leader_id', 'condition', 'model', 'framing', 'trait', 'score'])
    est = df.groupby(['leader_id', 'condition', 'model', 'framing', 'trait'], as_index=False)['score'].mean().rename(columns={'score': 'value'})
    est = _add_pooled(est, ['leader_id', 'condition', 'framing', 'trait'])
    tokens = {k: float(np.mean(v)) for k, v in tok.items()}
    return (est, tokens)

def _add_pooled(est: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pooled = est[est['model'].isin(OPEN_MODELS)].groupby(keys, as_index=False)['value'].mean()
    pooled['model'] = POOLED
    return pd.concat([est, pooled], ignore_index=True)

def est_dict(points: pd.DataFrame, model: str, framing: str, condition: str | None=None) -> dict[str, dict[str, float]]:
    sub = points[(points['model'] == model) & (points['framing'] == framing)]
    if condition is not None:
        sub = sub[sub['condition'] == condition]
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in sub.iterrows():
        out[r['leader_id']][r['trait']] = r['value']
    return out

def _present(v) -> bool:
    if v is None or v == '':
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True

def _rowwise_r(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(X) & np.isfinite(Y)
    with np.errstate(invalid='ignore', divide='ignore'):
        cnt = valid.sum(axis=1, keepdims=True)
        Xm = np.where(valid, X, 0.0).sum(axis=1, keepdims=True) / cnt
        Ym = np.where(valid, Y, 0.0).sum(axis=1, keepdims=True) / cnt
        Xc = np.where(valid, X - Xm, 0.0)
        Yc = np.where(valid, Y - Ym, 0.0)
        num = (Xc * Yc).sum(axis=1)
        den = np.sqrt((Xc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1))
        r = np.where(den > 0, num / den, np.nan)
    return r

def _aligned(est_by_leader, crit, traits):

    def pair(l, t):
        ev = est_by_leader.get(l, {}).get(t, None)
        cv = crit.get(l, {}).get(f'expert_{t}_mean', None)
        if _present(ev) and _present(cv):
            return (float(cv), float(ev))
        return (float('nan'), float('nan'))

    def ok(l):
        return any((pair(l, t)[0] == pair(l, t)[0] for t in traits))
    leaders = sorted((l for l in est_by_leader if ok(l)))
    out = {t: (np.array([pair(l, t)[0] for l in leaders]), np.array([pair(l, t)[1] for l in leaders])) for t in traits}
    return (out, leaders)

def aggregate_r(aligned, traits, ceilings):
    X = np.array([aligned[t][0] for t in traits], float)
    Y = np.array([aligned[t][1] for t in traits], float)
    cvec = np.array([ceilings[t] for t in traits], float)
    n = X.shape[1]
    ra = _rowwise_r(X, Y)
    out = {'r': float(np.nanmean(ra)), 'frac': float(np.nanmean(ra / cvec)), 'n': n}
    if n < 2:
        out.update(ci_low=float('nan'), ci_high=float('nan'), ci_low_frac=float('nan'), ci_high_frac=float('nan'))
        return out
    rng = np.random.default_rng(SEED)
    boot, boot_frac = (np.empty(N_BOOT), np.empty(N_BOOT))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        rb = _rowwise_r(X[:, idx], Y[:, idx])
        boot[b] = np.nanmean(rb)
        boot_frac[b] = np.nanmean(rb / cvec)
    out.update(ci_low=float(np.nanpercentile(boot, LO)), ci_high=float(np.nanpercentile(boot, HI)), ci_low_frac=float(np.nanpercentile(boot_frac, LO)), ci_high_frac=float(np.nanpercentile(boot_frac, HI)))
    return out

def converge_rows(est_by_leader, crit, ceilings, base):
    rows = []
    for t in TRAITS:
        aligned, _ = _aligned(est_by_leader, crit, (t,))
        x, y = aligned[t]
        pr = metrics.pearson(x, y, ci=True, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
        rows.append({**base, 'trait': t, 'n': pr.n, 'pearson_r': pr.statistic, 'ci_low': pr.ci_low, 'ci_high': pr.ci_high, 'ceiling': ceilings[t], 'r_frac_ceiling': pr.statistic / ceilings[t] if ceilings[t] else float('nan')})
    for name, traits in GROUPS.items():
        aligned, _ = _aligned(est_by_leader, crit, traits)
        ag = aggregate_r(aligned, traits, ceilings)
        ceil = float(np.mean([ceilings[t] for t in traits]))
        rows.append({**base, 'trait': name, 'n': ag['n'], 'pearson_r': ag['r'], 'ci_low': ag['ci_low'], 'ci_high': ag['ci_high'], 'ceiling': ceil, 'r_frac_ceiling': ag['frac'], 'ci_low_frac': ag['ci_low_frac'], 'ci_high_frac': ag['ci_high_frac']})
    return rows

def _triplet(a_ebl, b_ebl, crit, traits):

    def triple(l, t):
        av = a_ebl.get(l, {}).get(t, None)
        bv = b_ebl.get(l, {}).get(t, None)
        cv = crit.get(l, {}).get(f'expert_{t}_mean', None)
        if _present(av) and _present(bv) and _present(cv):
            return (float(cv), float(av), float(bv))
        return (float('nan'), float('nan'), float('nan'))
    leaders = sorted((l for l in set(a_ebl) & set(b_ebl) if any((triple(l, t)[0] == triple(l, t)[0] for t in traits))))
    E = {t: np.array([triple(l, t)[0] for l in leaders]) for t in traits}
    A = {t: np.array([triple(l, t)[1] for l in leaders]) for t in traits}
    B = {t: np.array([triple(l, t)[2] for l in leaders]) for t in traits}
    return (E, A, B, leaders)

def aggregate_paired(E, A, B, traits):
    Xe = np.array([E[t] for t in traits], float)
    Xa = np.array([A[t] for t in traits], float)
    Xb = np.array([B[t] for t in traits], float)
    n = Xe.shape[1]
    per_trait_diff = _rowwise_r(Xe, Xa) - _rowwise_r(Xe, Xb)
    r_a = float(np.nanmean(_rowwise_r(Xe, Xa)))
    r_b = float(np.nanmean(_rowwise_r(Xe, Xb)))
    diff = r_a - r_b
    if n < 3:
        return dict(r_a=r_a, r_b=r_b, diff=diff, ci_low=float('nan'), ci_high=float('nan'), p_gt0=float('nan'), n=n, n_neg_traits=int(np.sum(per_trait_diff < 0)))
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[i] = np.nanmean(_rowwise_r(Xe[:, idx], Xa[:, idx])) - np.nanmean(_rowwise_r(Xe[:, idx], Xb[:, idx]))
    return dict(r_a=r_a, r_b=r_b, diff=diff, ci_low=float(np.nanpercentile(boot, LO)), ci_high=float(np.nanpercentile(boot, HI)), p_gt0=float(np.mean(boot > 0)), n=n, n_neg_traits=int(np.sum(per_trait_diff < 0)))

def build_evidence_contrasts(namep, speechp, crit):
    rows = []
    models = [*OPEN_MODELS, POOLED, *ANCHORS]
    FRAMINGS = ('G', 'O', 'K')
    CONTRASTS = [('named-speech-name', 'named-speech', 'name'), ('named-speech-anon-speech', 'named-speech', 'anon-speech')]
    for framing in FRAMINGS:
        for model in models:
            named = est_dict(speechp, model, framing, 'named-speech')
            anon = est_dict(speechp, model, framing, 'anon-speech')
            name = est_dict(namep, model, framing)
            src = {'named-speech': named, 'anon-speech': anon, 'name': name}
            for label, ca, cb in CONTRASTS:
                a_ebl, b_ebl = (src[ca], src[cb])
                if not a_ebl or not b_ebl:
                    continue
                for t in TRAITS:
                    leaders = sorted(set(a_ebl) & set(b_ebl))
                    exp = np.array([crit.get(l, {}).get(f'expert_{t}_mean', np.nan) for l in leaders], float)
                    av = np.array([a_ebl[l].get(t, np.nan) for l in leaders], float)
                    bv = np.array([b_ebl[l].get(t, np.nan) for l in leaders], float)
                    cd = metrics.correlation_difference(exp, av, bv, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                    rows.append({'stratum': f'bridge:{framing}', 'model': model, 'trait': t, 'contrast': label, 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high, 'p_gt0': '', 'n_neg_traits': ''})
                for gname, traits in GROUPS.items():
                    E, A, B, _ = _triplet(a_ebl, b_ebl, crit, traits)
                    ag = aggregate_paired(E, A, B, traits)
                    rows.append({'stratum': f'bridge:{framing}', 'model': model, 'trait': gname, 'contrast': label, 'n': ag['n'], 'r_a': ag['r_a'], 'r_b': ag['r_b'], 'diff': ag['diff'], 'ci_low': ag['ci_low'], 'ci_high': ag['ci_high'], 'p_gt0': ag['p_gt0'], 'n_neg_traits': ag['n_neg_traits']})
    return pd.DataFrame(rows)

def build_speech_convergence(namep, speechp, crit, ceilings):
    rows = []
    models = [*OPEN_MODELS, POOLED, *ANCHORS]
    for cond, framings, pts, cfilter in [('name', ('G', 'O', 'K'), namep, None), ('named-speech', ('G', 'O', 'K'), speechp, 'named-speech'), ('anon-speech', ('G', 'O'), speechp, 'anon-speech')]:
        for framing in framings:
            for model in models:
                ebl = est_dict(pts, model, framing, cfilter)
                rows += converge_rows(ebl, crit, ceilings, {'condition': cond, 'framing': framing, 'model': model})
    return pd.DataFrame(rows)

def load_recognition():
    recog = {}
    recs = []
    for path in REID_FILES:
        for r in _jsonl(path):
            rd = r['reid'] or {}
            guess = f"{rd.get('name', '') or ''} {rd.get('country', '') or ''}".strip()
            label = score_reid_guess(guess, r['true_name'], country=r.get('true_country') or None)
            label = r['label']
            recog[r['speech_id'], r['model']] = label
            recs.append({'speech_id': r['speech_id'], 'model': r['model'], 'leader_id': r.get('name_arm_id'), 'in_name_arm': r['in_name_arm'], 'label': label})
    return (recog, recs)

def anon_speech_rows():
    rows = []
    for path in SPEECH_FILES:
        for r in _jsonl(path):
            if not r.get('parse_ok') or r['condition'] != 'anon-speech' or (not r.get('name_arm_id')):
                continue
            for trait, s in nsm.score_ratings(r['ratings']).items():
                rows.append({'leader_id': r['name_arm_id'], 'speech_id': r['speech_id'], 'model': r['model'], 'framing': r['framing'], 'trait': trait, 'score': s})
    return pd.DataFrame(rows)

def per_model_leader_est(anon, recog, model, framing, subset, recog_def):
    sub = anon[(anon['model'] == model) & (anon['framing'] == framing)]
    if subset != 'all':

        def keep(sp):
            lab = recog.get((sp, model), 'wrong')
            rec = lab == 'correct' if recog_def == 'strict' else lab in ('correct', 'partial')
            return rec if subset == 'recognized' else not rec
        sub = sub[sub['speech_id'].map(keep)]
    if sub.empty:
        return {}
    g = sub.groupby(['leader_id', 'trait'])['score'].mean()
    out = defaultdict(dict)
    for (leader, trait), v in g.items():
        out[leader][trait] = float(v)
    return out

def pool_leader_est(per_model):
    acc = defaultdict(lambda: defaultdict(list))
    for ebl in per_model.values():
        for leader, td in ebl.items():
            for trait, v in td.items():
                acc[leader][trait].append(v)
    return {l: {t: float(np.mean(vs)) for t, vs in td.items()} for l, td in acc.items()}

def build_anon_by_recognition(anon, recog, crit, ceilings):
    rows = []
    models = [*OPEN_MODELS, POOLED, *ANCHORS]
    for recog_def in ('strict', 'lenient'):
        for framing in ('G', 'O'):
            for subset in ('all', 'recognized', 'non_recognized'):
                per_model_ebl = {m: per_model_leader_est(anon, recog, m, framing, subset, recog_def) for m in (*OPEN_MODELS, *ANCHORS)}
                for model in models:
                    ebl = pool_leader_est({m: per_model_ebl[m] for m in OPEN_MODELS}) if model == POOLED else per_model_ebl[model]
                    base = {'recog_def': recog_def, 'framing': framing, 'subset': subset, 'model': model}
                    for gname, traits in GROUPS.items():
                        aligned, leaders = _aligned(ebl, crit, traits)
                        if leaders:
                            ag = aggregate_r(aligned, traits, ceilings)
                        else:
                            ag = {'r': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'frac': float('nan'), 'n': 0}
                        rows.append({**base, 'trait_group': gname, 'n_leaders': ag['n'], 'pearson_r': ag['r'], 'ci_low': ag['ci_low'], 'ci_high': ag['ci_high'], 'r_frac_ceiling': ag['frac']})
    return pd.DataFrame(rows)
SPEECH_CONDS = (('named-speech', ('G', 'O', 'K')), ('anon-speech', ('G', 'O')))

def speech_scored() -> pd.DataFrame:
    rows = []
    for path in SPEECH_FILES:
        for r in _jsonl(path):
            if not r.get('parse_ok') or not r.get('name_arm_id'):
                continue
            for trait, s in nsm.score_ratings(r['ratings']).items():
                rows.append((r['speech_id'], r['condition'], r['model'], r['framing'], r['repeat'], trait, s))
    return pd.DataFrame(rows, columns=['speech_id', 'condition', 'model', 'framing', 'repeat', 'trait', 'score'])

def build_speech_reliability(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, framings in SPEECH_CONDS:
        for model in (*OPEN_MODELS, *ANCHORS):
            for framing in framings:
                for trait in TRAITS:
                    cell = scored[(scored['condition'] == cond) & (scored['model'] == model) & (scored['framing'] == framing) & (scored['trait'] == trait)]
                    mat = cell.pivot(index='speech_id', columns='repeat', values='score').dropna(axis=0)
                    arr = mat.to_numpy(float)
                    icc11 = nsm.icc_1_1(arr)
                    icc21 = metrics.icc(arr).icc1 if arr.shape[0] >= 2 and arr.shape[1] >= 2 else float('nan')
                    rows.append({'condition': cond, 'model': model, 'framing': framing, 'trait': trait, 'n_subjects': arr.shape[0], 'n_repeats': arr.shape[1], 'icc_1_1': icc11, 'icc_2_1_crosscheck': icc21})
    return pd.DataFrame(rows)

def build_speech_bias(speechp: pd.DataFrame, crit) -> pd.DataFrame:
    rows = []
    for cond, framings in SPEECH_CONDS:
        for model in (*OPEN_MODELS, POOLED, *ANCHORS):
            for framing in framings:
                for trait in TRAITS:
                    sub = speechp[(speechp['model'] == model) & (speechp['condition'] == cond) & (speechp['framing'] == framing) & (speechp['trait'] == trait)]
                    raw_gaps, sd_gaps = ([], [])
                    for _, row in sub.iterrows():
                        c = crit.get(row['leader_id'], {})
                        ev = c.get(f'expert_{trait}_mean')
                        esd = c.get(f'expert_{trait}_sd')
                        if not _present(ev) or not _present(row['value']):
                            continue
                        b = metrics.directional_bias(float(row['value']), float(ev), None if not _present(esd) else float(esd))
                        raw_gaps.append(b.raw_gap)
                        if b.sd_units is not None:
                            sd_gaps.append(b.sd_units)
                    rows.append({'condition': cond, 'model': model, 'framing': framing, 'trait': trait, 'n_raw': len(raw_gaps), 'n_sd': len(sd_gaps), 'bias_raw_mean': float(np.mean(raw_gaps)) if raw_gaps else float('nan'), 'bias_sd_units_mean': float(np.mean(sd_gaps)) if sd_gaps else float('nan')})
    return pd.DataFrame(rows)

def build_speech_framing_contrasts(speechp: pd.DataFrame, crit) -> pd.DataFrame:
    rows = []
    CONTRASTS = {'named-speech': [('K-G', 'K', 'G'), ('O-G', 'O', 'G'), ('K-O', 'K', 'O')], 'anon-speech': [('O-G', 'O', 'G')]}
    for cond, contrasts in CONTRASTS.items():
        for model in (*OPEN_MODELS, POOLED, *ANCHORS):
            for label, fa, fb in contrasts:
                a_ebl = est_dict(speechp, model, fa, cond)
                b_ebl = est_dict(speechp, model, fb, cond)
                if not a_ebl or not b_ebl:
                    continue
                for t in TRAITS:
                    leaders = sorted(set(a_ebl) & set(b_ebl))
                    exp = np.array([crit.get(l, {}).get(f'expert_{t}_mean', np.nan) for l in leaders], float)
                    av = np.array([a_ebl[l].get(t, np.nan) for l in leaders], float)
                    bv = np.array([b_ebl[l].get(t, np.nan) for l in leaders], float)
                    cd = metrics.correlation_difference(exp, av, bv, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                    rows.append({'condition': cond, 'model': model, 'trait': t, 'contrast': label, 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high, 'p_gt0': '', 'n_neg_traits': ''})
                for gname, traits in GROUPS.items():
                    E, A, B, _ = _triplet(a_ebl, b_ebl, crit, traits)
                    ag = aggregate_paired(E, A, B, traits)
                    rows.append({'condition': cond, 'model': model, 'trait': gname, 'contrast': label, 'n': ag['n'], 'r_a': ag['r_a'], 'r_b': ag['r_b'], 'diff': ag['diff'], 'ci_low': ag['ci_low'], 'ci_high': ag['ci_high'], 'p_gt0': ag['p_gt0'], 'n_neg_traits': ag['n_neg_traits']})
    return pd.DataFrame(rows)
