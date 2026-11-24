# Adapted from the executed study code; see docs/provenance.md.
from __future__ import annotations
import warnings
from typing import Callable, NamedTuple, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, norm
from . import config
from .schema import ExpertScore, ManifestRow, RatingRecord
__all__ = ['Correlation', 'CorrelationDiff', 'Bias', 'ICCResult', 'pearson', 'spearman', 'bootstrap_ci', 'correlation_difference', 'overlap', 'overlap_normals', 'overlap_equal_var_closed_form', 'directional_bias', 'icc', 'within_model_reliability', 'inter_model_agreement', 'agreement_table']

class Correlation(NamedTuple):
    statistic: float
    pvalue: float
    n: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

class CorrelationDiff(NamedTuple):
    diff: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    r_a: float
    r_b: float
    n: int

class Bias(NamedTuple):
    raw_gap: float
    sd_units: Optional[float]

class ICCResult(NamedTuple):
    icc1: float
    icck: float
    n_subjects: int
    n_raters: int

def _clean_pair(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return (x[mask], y[mask])

def _safe_stat(fn: Callable, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(fn(x, y).statistic)

def _pearson_r_fast(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return float('nan')
    return float(xm.dot(ym) / denom)

def bootstrap_ci(x: Sequence[float], y: Sequence[float], statistic: Callable[[np.ndarray, np.ndarray], float], *, n_boot: int=2000, ci: float=0.95, seed: int=0) -> tuple[float, float]:
    x, y = _clean_pair(x, y)
    n = x.size
    if n < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = statistic(x[idx], y[idx])
    lo = float(np.nanpercentile(boot, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.nanpercentile(boot, (1.0 + ci) / 2.0 * 100.0))
    return (lo, hi)

def _correlate(fn: Callable, x: Sequence[float], y: Sequence[float], *, ci: bool, n_boot: int, ci_level: float, seed: int, boot_stat: Optional[Callable[[np.ndarray, np.ndarray], float]]=None) -> Correlation:
    x, y = _clean_pair(x, y)
    n = int(x.size)
    if n < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return Correlation(float('nan'), float('nan'), n, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fn(x, y)
    lo = hi = None
    if ci:
        stat = boot_stat if boot_stat is not None else lambda a, b: _safe_stat(fn, a, b)
        lo, hi = bootstrap_ci(x, y, stat, n_boot=n_boot, ci=ci_level, seed=seed)
    return Correlation(float(res.statistic), float(res.pvalue), n, lo, hi)

def pearson(x: Sequence[float], y: Sequence[float], *, ci: bool=False, n_boot: int=2000, ci_level: float=0.95, seed: int=0) -> Correlation:
    return _correlate(stats.pearsonr, x, y, ci=ci, n_boot=n_boot, ci_level=ci_level, seed=seed, boot_stat=_pearson_r_fast)

def spearman(x: Sequence[float], y: Sequence[float], *, ci: bool=False, n_boot: int=2000, ci_level: float=0.95, seed: int=0) -> Correlation:
    return _correlate(stats.spearmanr, x, y, ci=ci, n_boot=n_boot, ci_level=ci_level, seed=seed)

def correlation_difference(criterion: Sequence[float], series_a: Sequence[float], series_b: Sequence[float], *, n_boot: int=2000, ci_level: float=0.95, seed: int=0) -> CorrelationDiff:
    c = np.asarray(criterion, dtype=float)
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    mask = np.isfinite(c) & np.isfinite(a) & np.isfinite(b)
    c, a, b = (c[mask], a[mask], b[mask])
    n = int(c.size)
    if n < 3 or np.ptp(c) == 0:
        return CorrelationDiff(float('nan'), None, None, float('nan'), float('nan'), n)
    r_a = _safe_stat(stats.pearsonr, c, a)
    r_b = _safe_stat(stats.pearsonr, c, b)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ci_, ai_, bi_ = (c[idx], a[idx], b[idx])
        if np.ptp(ci_) == 0:
            boot[i] = float('nan')
            continue
        boot[i] = _pearson_r_fast(ci_, ai_) - _pearson_r_fast(ci_, bi_)
    lo = float(np.nanpercentile(boot, (1.0 - ci_level) / 2.0 * 100.0))
    hi = float(np.nanpercentile(boot, (1.0 + ci_level) / 2.0 * 100.0))
    return CorrelationDiff(float(r_a - r_b), lo, hi, float(r_a), float(r_b), n)

def overlap_equal_var_closed_form(mu1: float, mu2: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError('sigma must be > 0')
    return float(2.0 * norm.cdf(-abs(mu1 - mu2) / (2.0 * sigma)))

def overlap_normals(mu1: float, sd1: float, mu2: float, sd2: float, *, grid: int=20001, span: float=12.0) -> float:
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError('standard deviations must be > 0')
    lo = min(mu1 - span * sd1, mu2 - span * sd2)
    hi = max(mu1 + span * sd1, mu2 + span * sd2)
    xs = np.linspace(lo, hi, grid)
    f1 = norm.pdf(xs, mu1, sd1)
    f2 = norm.pdf(xs, mu2, sd2)
    return float(np.trapezoid(np.minimum(f1, f2), xs))

def overlap(expert_mean: float, expert_sd: Optional[float], llm_draws: Sequence[float], *, method: str='empirical', grid: int=20001, span: float=12.0, llm_sd_floor: float=1e-06) -> Optional[float]:
    if expert_sd is None or expert_sd == 0:
        return None
    draws = np.asarray(llm_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    if method not in ('empirical', 'normal'):
        raise ValueError(f"unknown method {method!r} (use 'empirical' or 'normal')")
    degenerate = draws.size < 2 or np.ptp(draws) == 0
    if method == 'normal' or degenerate:
        mu_l = float(draws.mean())
        sd_l = float(draws.std(ddof=1)) if draws.size > 1 else 0.0
        sd_l = max(sd_l, llm_sd_floor)
        return overlap_normals(expert_mean, float(expert_sd), mu_l, sd_l, grid=grid, span=span)
    kde = gaussian_kde(draws)
    draw_sd = float(draws.std(ddof=0))
    lo = min(expert_mean - span * expert_sd, draws.min() - span * draw_sd)
    hi = max(expert_mean + span * expert_sd, draws.max() + span * draw_sd)
    xs = np.linspace(lo, hi, grid)
    f_expert = norm.pdf(xs, expert_mean, float(expert_sd))
    f_llm = kde(xs)
    return float(np.trapezoid(np.minimum(f_expert, f_llm), xs))

def directional_bias(llm_mean: float, expert_mean: float, expert_sd: Optional[float]) -> Bias:
    raw_gap = float(llm_mean) - float(expert_mean)
    if expert_sd is None or expert_sd == 0:
        return Bias(raw_gap, None)
    return Bias(raw_gap, raw_gap / float(expert_sd))
_LONG_COLS = ['leader_id', 'condition', 'model', 'occasion', 'repeat', 'trait', 'score']

def _records_long(records: Sequence[RatingRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        for trait, score in r.trait_scores.items():
            rows.append((r.leader_id, r.condition, r.model, r.occasion, r.repeat, trait, float(score)))
    return pd.DataFrame(rows, columns=_LONG_COLS)

def _scale_var_max() -> float:
    rng = config.SCALE_MAX - config.SCALE_MIN
    return (rng / 2.0) ** 2

def within_model_reliability(records: Sequence[RatingRecord], *, var_max: Optional[float]=None) -> pd.DataFrame:
    if var_max is None:
        var_max = _scale_var_max()
    long = _records_long(records)
    cols = ['model', 'condition', 'leader_id', 'occasion', 'trait']
    out_cols = cols + ['n_repeats', 'var', 'consistency']
    if long.empty:
        return pd.DataFrame(columns=out_cols)
    grp = long.groupby(cols, dropna=False)['score']
    out = grp.agg(n_repeats='size').reset_index()
    var = grp.var(ddof=0).reset_index(name='var')
    out = out.merge(var, on=cols)
    out['consistency'] = (1.0 - out['var'] / var_max).clip(lower=0.0, upper=1.0)
    return out[out_cols].sort_values(cols, ignore_index=True)

def icc(matrix: np.ndarray) -> ICCResult:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2 or (not np.isfinite(m).all()):
        return ICCResult(float('nan'), float('nan'), int(m.shape[0]), int(m.shape[1]))
    n, k = m.shape
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((m - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_err / ((n - 1) * (k - 1))
    denom1 = ms_r + (k - 1) * ms_e + k / n * (ms_c - ms_e)
    denomk = ms_r + (ms_c - ms_e) / n
    icc1 = (ms_r - ms_e) / denom1 if denom1 != 0 else float('nan')
    icck = (ms_r - ms_e) / denomk if denomk != 0 else float('nan')
    return ICCResult(float(icc1), float(icck), n, k)

def _mean_pairwise_pearson(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    if k < 2:
        return float('nan')
    rs = []
    for i in range(k):
        for j in range(i + 1, k):
            r = _safe_stat(stats.pearsonr, matrix[:, i], matrix[:, j])
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else float('nan')

def inter_model_agreement(records: Sequence[RatingRecord], *, condition: Optional[str]=None) -> pd.DataFrame:
    out_cols = ['condition', 'trait', 'n_leaders', 'n_models', 'icc2_1', 'icc2_k', 'mean_pairwise_r']
    long = _records_long(records)
    if condition is not None:
        long = long[long['condition'] == condition]
    if long.empty:
        return pd.DataFrame(columns=out_cols)
    leader_level = long.groupby(['condition', 'model', 'leader_id', 'trait'], dropna=False)['score'].mean().reset_index()
    rows = []
    for (cond, trait), sub in leader_level.groupby(['condition', 'trait']):
        mat = sub.pivot(index='leader_id', columns='model', values='score')
        mat = mat.dropna(axis=0)
        n_leaders, n_models = mat.shape
        if n_leaders >= 2 and n_models >= 2:
            ic = icc(mat.to_numpy())
            mpr = _mean_pairwise_pearson(mat.to_numpy())
            rows.append((cond, trait, n_leaders, n_models, ic.icc1, ic.icck, mpr))
        else:
            rows.append((cond, trait, n_leaders, n_models, float('nan'), float('nan'), float('nan')))
    return pd.DataFrame(rows, columns=out_cols).sort_values(['condition', 'trait'], ignore_index=True)
_TABLE_COLS = ['trait', 'n_leaders', 'pearson_r', 'pearson_p', 'pearson_ci_low', 'pearson_ci_high', 'spearman_rho', 'spearman_p', 'overlap_mean_pct', 'n_overlap', 'bias_raw_mean', 'bias_sd_units_mean']

def agreement_table(records: Sequence[RatingRecord], manifest: Sequence[ManifestRow], condition: str, *, model: Optional[str]=None, overlap_method: str='empirical', ci: bool=False, n_boot: int=2000, ci_level: float=0.95, seed: int=0) -> pd.DataFrame:
    by_id: dict[str, ManifestRow] = {m.leader_id: m for m in manifest}
    long = _records_long(records)
    long = long[long['condition'] == condition]
    if model is not None:
        long = long[long['model'] == model]
    rows = []
    for trait in config.TRAITS:
        sub = long[long['trait'] == trait]
        entries: list[tuple[str, ExpertScore, np.ndarray, float]] = []
        if not sub.empty:
            for leader_id, g in sub.groupby('leader_id'):
                row = by_id.get(leader_id)
                if row is None:
                    continue
                expert = row.expert.get(trait)
                if expert is None:
                    continue
                draws = g['score'].to_numpy(dtype=float)
                if draws.size == 0:
                    continue
                entries.append((leader_id, expert, draws, float(draws.mean())))
        if not entries:
            rows.append((trait, 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0, float('nan'), float('nan')))
            continue
        expert_means = np.array([e.mean for _, e, _, _ in entries], dtype=float)
        llm_means = np.array([mu for _, _, _, mu in entries], dtype=float)
        pr = pearson(expert_means, llm_means, ci=ci, n_boot=n_boot, ci_level=ci_level, seed=seed)
        sr = spearman(expert_means, llm_means)
        overlaps = []
        sd_unit_gaps = []
        for _, expert, draws, mu in entries:
            ov = overlap(expert.mean, expert.sd, draws, method=overlap_method)
            if ov is not None:
                overlaps.append(ov)
            bias = directional_bias(mu, expert.mean, expert.sd)
            if bias.sd_units is not None:
                sd_unit_gaps.append(bias.sd_units)
        overlap_pct = float(np.mean(overlaps)) * 100.0 if overlaps else float('nan')
        bias_raw = float(np.mean(llm_means - expert_means))
        bias_sd = float(np.mean(sd_unit_gaps)) if sd_unit_gaps else float('nan')
        rows.append((trait, len(entries), pr.statistic, pr.pvalue, pr.ci_low, pr.ci_high, sr.statistic, sr.pvalue, overlap_pct, len(overlaps), bias_raw, bias_sd))
    return pd.DataFrame(rows, columns=_TABLE_COLS)
