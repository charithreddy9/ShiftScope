import numpy as np
import pytest
from sklearn.datasets import load_digits
from shiftscope.experiment import split_data,corrupt,run
from shiftscope.metrics import apply_temperature,fit_temperature,conformal_threshold

def test_no_leakage():
    y=load_digits().target
    s=split_data(y,7)
    combined=np.concatenate([s.train,s.temperature,s.conformal,s.test])
    assert len(combined)==len(y)==len(set(combined))
    assert abs(len(s.train)/len(y)-.6)<.01

def test_corruption_is_repeatable():
    x=load_digits().data[:8]
    assert np.array_equal(corrupt(x,2,5),corrupt(x,2,5))
    assert np.array_equal(corrupt(x,0,5),x)
    assert np.min(corrupt(x,2,5))>=0

def test_temperature_normalizes():
    p=np.array([[.8,.2],[.3,.7]])
    out=apply_temperature(p,2)
    assert np.allclose(out.sum(axis=1),1)
    assert np.array_equal(out.argmax(axis=1),p.argmax(axis=1))
    with pytest.raises(ValueError):apply_temperature(p,0)

def test_finite_sample_conformal():
    p=np.tile([.8,.2],(9,1))
    assert conformal_threshold(p,np.zeros(9,dtype=int),.2)==pytest.approx(.2)

def test_short_experiment():
    rows=run((7,),(0.,))
    assert len(rows)==4
    assert all(0<=x["accuracy"]<=1 for x in rows)
