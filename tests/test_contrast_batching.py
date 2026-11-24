"""Batching must retain the original resampling worlds and statistical definition."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import speech_model_contrasts as sc
from pinf import metrics


@pytest.mark.parametrize('n',[7,33,50])
def test_paired_trait_matches_executed_scalar_bootstrap(n):
    rng=np.random.default_rng(91)
    c,a,b=rng.normal(size=(3,n));a[1]=np.nan
    expected=metrics.correlation_difference(c,a,b,n_boot=120,seed=73)
    actual=sc._paired_trait(c,a,b,n_boot=120,seed=73)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12,equal_nan=True)


def test_aggregate_batching_matches_scalar_worlds_with_missing_traits(monkeypatch):
    rng=np.random.default_rng(14)
    e,a,b=rng.normal(size=(3,3,17));e[1,3]=np.nan;a[2,5]=np.nan
    c=np.array([.8,.9,.7]);monkeypatch.setattr(sc,'N_BOOT',100)
    r=np.random.default_rng(sc.SEED);raw=[];fraction=[]
    for _ in range(100):
        ix=r.integers(0,17,17)
        ra=sc.sm._rowwise_r(e[:,ix],a[:,ix]);rb=sc.sm._rowwise_r(e[:,ix],b[:,ix])
        raw.append(np.nanmean(ra)-np.nanmean(rb))
        fraction.append(np.nanmean(ra/c)-np.nanmean(rb/c))
    actual=sc._aggregate_paired_models(e,a,b,c)
    np.testing.assert_allclose([actual['ci_low'],actual['ci_high']],np.nanpercentile(raw,[sc.LO,sc.HI]),atol=1e-12)
    np.testing.assert_allclose([actual['ci_low_frac'],actual['ci_high_frac']],np.nanpercentile(fraction,[sc.LO,sc.HI]),atol=1e-12)
    expected_p=min(1.,2*min(np.mean(np.asarray(fraction)<=0),np.mean(np.asarray(fraction)>=0)))
    assert actual['p_two_sided']==expected_p
