"""Independent checks of scoring and statistical conventions, not implementation snapshots."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import name_sweep_metrics as nsm
import contrast_metrics as cm
import speech_metrics as sm
from pinf import metrics


def test_reverse_keying_all_trait_extremes():
    r={f's{i}':4 for i in range(1,17)}
    for i in (2,4,6,8,10):r[f's{i}']=0
    assert set(nsm.score_ratings(r).values())=={4.0}
    opposite={k:4-v for k,v in r.items()}
    assert set(nsm.score_ratings(opposite).values())=={0.0}


def test_open_ensemble_averages_ratings_not_correlations():
    rows=[]
    values={'qwen-8b':0.,'qwen-32b':1.,'gemma-12b':2.,'gemma-27b':3.,'gpt-5.5':4.}
    for model,value in values.items():
        for repeat in range(3):
            rows.append(('person',model,'O',repeat,'openness',value))
    points=nsm.point_estimates(pd.DataFrame(rows,columns=['leader_id','model','framing','repeat','trait','score']))
    assert points.loc[points.model=='pooled','llm_mean'].item()==1.5


def test_one_way_icc_perfect_repeatability_and_between_subject_variation():
    # An exact hand-calculable balanced one-way ANOVA example.
    matrix=np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
    assert nsm.icc_1_1(matrix)==pytest.approx(1.)


def test_paired_bootstrap_self_difference_is_exactly_zero():
    x=np.array([0.,1.,2.,3.,4.,5.])
    y=np.array([1.,0.,2.,4.,3.,5.])
    r=metrics.correlation_difference(x,y,y,n_boot=100,seed=7)
    assert r.diff==pytest.approx(0)
    assert r.ci_low==pytest.approx(0)
    assert r.ci_high==pytest.approx(0)


def test_missing_criterion_is_trait_specific():
    crit={'a':{'expert_extraversion_mean':1.,'expert_openness_mean':1.},
          'b':{'expert_extraversion_mean':2.,'expert_openness_mean':np.nan},
          'c':{'expert_extraversion_mean':3.,'expert_openness_mean':3.}}
    est={k:{'extraversion':v,'openness':v} for k,v in zip(crit,[1.,2.,3.])}
    aligned,leaders=sm._aligned(est,crit,['extraversion','openness'])
    assert len(leaders)==3
    assert np.isfinite(aligned['extraversion'][0]).sum()==3
    assert np.isfinite(aligned['openness'][0]).sum()==2


def test_mean_fraction_ceiling_uses_trait_specific_denominators():
    x=np.array([np.arange(20.),np.arange(20.)])
    result=cm._aggregate_level(x,x,np.array([.5,1.]))
    assert result['mean_r']==pytest.approx(1.)
    assert result['mean_frac_ceiling']==pytest.approx(1.5)
