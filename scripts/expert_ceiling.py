import json
import numpy as np
import pandas as pd
DTA = 'data/NEGex_2.0/NEGex_2.0_candidates.dta'
SCALE = 4
N_SOURCE = 'recovered'
BFI_ITEMS = [f'bfi_c{i}' for i in range(1, 11)]
TRIAD_ITEMS = [f'triad_c{i}' for i in range(1, 7)]
BFI_REVERSE = {'bfi_c2', 'bfi_c4', 'bfi_c6', 'bfi_c8', 'bfi_c10'}
TRAIT_AGG = {'extraversion': ('bfi_c1', 'bfi_c6'), 'agreeableness': ('bfi_c7', 'bfi_c2'), 'conscientiousness': ('bfi_c3', 'bfi_c8'), 'emotional_stability': ('bfi_c9', 'bfi_c4'), 'openness': ('bfi_c5', 'bfi_c10'), 'narcissism': ('triad_c1', 'triad_c4'), 'psychopathy': ('triad_c2', 'triad_c5'), 'machiavellianism': ('triad_c3', 'triad_c6')}
BIG_FIVE = ['extraversion', 'agreeableness', 'conscientiousness', 'emotional_stability', 'openness']
DARK_TRIAD = ['narcissism', 'psychopathy', 'machiavellianism']

def _feasible(n, S, Q, scale=SCALE):
    if S < 0 or Q < 0:
        return False
    for c4 in range(0, n + 1):
        for c3 in range(0, n - c4 + 1):
            A = S - 3 * c3 - 4 * c4
            B = Q - 9 * c3 - 16 * c4
            if (B - A) % 2 != 0:
                continue
            c2 = (B - A) // 2
            c1 = A - 2 * c2
            c0 = n - c1 - c2 - c3 - c4
            if c0 >= 0 and c1 >= 0 and (c2 >= 0):
                return True
    return False

def recover_n(mean, sd, nmax=60):
    if not np.isfinite(sd):
        return None
    if sd == 0:
        return None
    for n in range(2, nmax + 1):
        S = mean * n
        if abs(S - round(S)) > 0.01:
            continue
        S = int(round(S))
        Q = (n - 1) * sd * sd + n * mean * mean
        if abs(Q - round(Q)) > 0.06:
            continue
        Q = int(round(Q))
        if _feasible(n, S, Q):
            return n
    return None

def oneway_icc(B, n, s):
    B = np.asarray(B, float)
    n = np.asarray(n, float)
    s = np.asarray(s, float)
    J = len(B)
    N = n.sum()
    grand = (n * B).sum() / N
    MSB = (n * (B - grand) ** 2).sum() / (J - 1)
    MSW = ((n - 1) * s ** 2).sum() / (n - 1).sum()
    n0 = (N - (n ** 2).sum() / N) / (J - 1)
    tau2 = (MSB - MSW) / n0
    icc1 = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
    return dict(J=J, N=int(N), MSB=MSB, MSW=MSW, sigma2=MSW, n0=n0, tau2=tau2, tau2_trunc=max(tau2, 0.0), icc1=icc1)

def reml_tau2(y, v):
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    hi = max(np.var(y) * 3.0, float(np.mean(v)) * 3.0, 0.0001)
    grid = np.linspace(0.0, hi, 6000)
    best, bestval = (0.0, np.inf)
    for t2 in grid:
        V = v + t2
        w = 1.0 / V
        mu = (w * y).sum() / w.sum()
        nll = 0.5 * (np.log(V).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
        if nll < bestval:
            bestval, best = (nll, t2)
    return best

def sb(icc1, k):
    return k * icc1 / (1 + (k - 1) * icc1)
df = pd.read_stata(DTA)
items = BFI_ITEMS + TRIAD_ITEMS
percell = {}
for it in items:
    mean = df[it].astype(float)
    sd = df[f'sd{it}'].astype(float)
    present = mean.notna()
    n = []
    for m, s, p in zip(mean, sd, present):
        if not p:
            n.append(np.nan)
        elif not np.isfinite(s):
            n.append(1)
        elif s == 0:
            n.append(0)
        else:
            r = recover_n(m, s)
            n.append(r if r is not None else np.nan)
    percell[it] = pd.DataFrame({'mean': mean, 'sd': sd, 'n': n, 'present': present})

def battery_counts(item_list):
    M = pd.DataFrame({it: percell[it]['n'] for it in item_list})
    anyrow = M.notna().any(axis=1) & (M.fillna(0) > 0).any(axis=1) | M.eq(1).any(axis=1)
    rows = M[M.apply(lambda r: r.notna().any(), axis=1)]
    present_rows = pd.DataFrame({it: percell[it]['present'] for it in item_list}).any(axis=1)
    sub = M[present_rows]
    batt = sub.replace(0, np.nan).max(axis=1)
    single = batt == 1
    return (sub, batt, single)
bfi_sub, bfi_batt, bfi_single = battery_counts(BFI_ITEMS)
desc = {'n_personality_rows_bfi': int(pd.DataFrame({it: percell[it]['present'] for it in BFI_ITEMS}).any(axis=1).sum()), 'battery_rater_count_median': float(np.nanmedian(bfi_batt)), 'battery_rater_count_min': float(np.nanmin(bfi_batt)), 'battery_rater_count_max': float(np.nanmax(bfi_batt)), 'single_rater_share_bfi': float(np.nanmean(bfi_single.astype(float)))}
peritem_n = {}
for it in items:
    nn = percell[it]['n']
    valid = nn[nn >= 2]
    peritem_n[it] = dict(median=float(valid.median()), min=int(valid.min()), max=int(valid.max()), n_leaders_ge2=int((nn >= 2).sum()), n_single=int((nn == 1).sum()), n_sd0=int((nn == 0).sum()))
item_res = {}
for it in items:
    d = percell[it]
    use = d[(d['n'] >= 2) & np.isfinite(d['sd']) & (d['sd'] > 0)].copy()
    B, n, s = (use['mean'].values, use['n'].values, use['sd'].values)
    r = oneway_icc(B, n, s)
    v = r['sigma2'] / n
    r['tau2_reml'] = reml_tau2(B, v)
    r['icc1_reml'] = r['tau2_reml'] / (r['tau2_reml'] + r['sigma2'])
    kmed = float(np.median(n))
    r['k_med'] = kmed
    r['icc1k'] = sb(r['icc1'], kmed)
    r['ceiling'] = np.sqrt(max(r['icc1k'], 0.0))
    r['sd_of_means'] = float(np.std(B, ddof=1))
    item_res[it] = r

def trait_reliability(a, b, rho_list=(0.0, 0.3, 0.5)):
    da, db = (percell[a], percell[b])
    mask = (da['n'] >= 2) & (da['sd'] > 0) & np.isfinite(da['sd']) & (db['n'] >= 2) & (db['sd'] > 0) & np.isfinite(db['sd'])
    A, Bb = (da[mask], db[mask])
    ga = SCALE - A['mean'].values if a in BFI_REVERSE else A['mean'].values
    gb = SCALE - Bb['mean'].values if b in BFI_REVERSE else Bb['mean'].values
    na, nb = (A['n'].values, Bb['n'].values)
    s2a, s2b = (item_res[a]['sigma2'], item_res[b]['sigma2'])
    t2a, t2b = (item_res[a]['tau2_trunc'], item_res[b]['tau2_trunc'])
    sa, sb_ = (np.sqrt(s2a), np.sqrt(s2b))
    emp_cov = np.cov(ga, gb, ddof=1)[0, 1]
    mean_inv_n = float(np.mean(1.0 / ((na + nb) / 2.0)))
    kmed = float(np.median((na + nb) / 2.0))
    out = {'J': int(mask.sum()), 'k_med': kmed, 'emp_cov': emp_cov, 'sd_of_trait_means': float(np.std((ga + gb) / 2.0, ddof=1))}
    by_rho = {}
    for rho in rho_list:
        tau_ab = emp_cov - rho * sa * sb_ * mean_inv_n
        true_var = (t2a + t2b + 2 * tau_ab) / 4.0
        err_k = (s2a + s2b + 2 * rho * sa * sb_) / (4.0 * kmed)
        err_1 = (s2a + s2b + 2 * rho * sa * sb_) / 4.0
        icc1k = true_var / (true_var + err_k)
        icc11 = true_var / (true_var + err_1)
        by_rho[rho] = dict(icc11=icc11, icc1k=icc1k, ceiling=np.sqrt(max(icc1k, 0.0)), tau_ab=tau_ab, true_var=true_var)
    out['by_rho'] = by_rho
    return out
trait_res = {t: trait_reliability(*pair) for t, pair in TRAIT_AGG.items()}

def fmt(x, d=3):
    return f'{x:.{d}f}'
print('N source:', N_SOURCE)
print('\n=== 1. DESCRIPTIVES (Big Five battery) ===')
for k, v in desc.items():
    print(f'  {k}: {v}')
print('  per-item recovered-n (median / range / #leaders n>=2 / #single / #sd0):')
for it in items:
    p = peritem_n[it]
    print(f"    {it:10} med {p['median']:.0f}  range {p['min']}-{p['max']}  J>=2 {p['n_leaders_ge2']:3}  single {p['n_single']:3}  sd0 {p['n_sd0']:3}")
print('\n=== 2. PER-ITEM ICC ===')
print(f"  {'item':10} {'J':>4} {'k':>4} {'ICC(1,1)':>9} {'ICC1_REML':>9} {'ICC(1,k)':>9} {'ceiling':>8} {'SDmeans':>8}")
for it in items:
    r = item_res[it]
    print(f"  {it:10} {r['J']:4d} {r['k_med']:4.0f} {fmt(r['icc1']):>9} {fmt(r['icc1_reml']):>9} {fmt(r['icc1k']):>9} {fmt(r['ceiling']):>8} {fmt(r['sd_of_means']):>8}")
print('\n=== 3. PER-TRAIT (rho_w sensitivity) ===')
print(f"  {'trait':20} {'J':>4} {'k':>4} | ICC(1,k) @rho 0/0.3/0.5  |  ceiling @rho 0/0.3/0.5  | SDtraitmeans")
for fam, traits in (('BIG FIVE', BIG_FIVE), ('DARK TRIAD', DARK_TRIAD)):
    print(f'  -- {fam} --')
    for t in traits:
        r = trait_res[t]
        br = r['by_rho']
        icc = '/'.join((fmt(br[x]['icc1k'], 2) for x in (0.0, 0.3, 0.5)))
        cei = '/'.join((fmt(br[x]['ceiling'], 2) for x in (0.0, 0.3, 0.5)))
        print(f"  {t:20} {r['J']:4d} {r['k_med']:4.0f} | {icc:>20} | {cei:>20} | {fmt(r['sd_of_trait_means'], 2)}")
print('\n=== 4. PER-FAMILY SUMMARY (rho_w=0.3) ===')
for fam, traits in (('Big Five', BIG_FIVE), ('Dark Triad', DARK_TRIAD)):
    iccs = [trait_res[t]['by_rho'][0.3]['icc1k'] for t in traits]
    ceis = [trait_res[t]['by_rho'][0.3]['ceiling'] for t in traits]
    print(f'  {fam:11}: ICC(1,k) median {fmt(np.median(iccs), 2)} range {fmt(min(iccs), 2)}-{fmt(max(iccs), 2)} | ceiling median {fmt(np.median(ceis), 2)} range {fmt(min(ceis), 2)}-{fmt(max(ceis), 2)}')
out = {'n_source': N_SOURCE, 'descriptives': desc, 'peritem_n': peritem_n, 'items': {it: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in item_res[it].items()} for it in items}, 'traits': {t: {'J': trait_res[t]['J'], 'k_med': trait_res[t]['k_med'], 'sd_of_trait_means': trait_res[t]['sd_of_trait_means'], 'by_rho': {str(rho): {k: float(vv) for k, vv in d.items()} for rho, d in trait_res[t]['by_rho'].items()}} for t in TRAIT_AGG}}
with open('outputs/expert_ceiling.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nwrote outputs/expert_ceiling.json')
