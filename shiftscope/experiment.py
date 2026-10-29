"""CPU-only, stratified evaluation on the built-in scikit-learn digits data."""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .metrics import apply_temperature,fit_temperature,conformal_threshold,prediction_sets,evaluate

SEEDS=(7,21,42)
SIGMAS=(0.,1.5,3.,4.5)

@dataclass(frozen=True)
class Split:
    train:np.ndarray
    temperature:np.ndarray
    conformal:np.ndarray
    test:np.ndarray

def split_data(y,seed):
    idx=np.arange(len(y))
    train_and_cal,test=train_test_split(idx,test_size=.2,stratify=y,random_state=seed)
    train,cal=train_test_split(train_and_cal,test_size=.25,stratify=y[train_and_cal],random_state=seed+10000)
    temp,conformal=train_test_split(cal,test_size=.5,stratify=y[cal],random_state=seed+20000)
    return Split(train,temp,conformal,test)

def corrupt(X,sigma,seed):
    if sigma<0: raise ValueError("Noise must be nonnegative")
    if sigma==0:return np.asarray(X,float).copy()
    return np.clip(X+np.random.default_rng(seed).normal(0,sigma,X.shape),0,16)

def run(seeds=SEEDS,sigmas=SIGMAS):
    data=load_digits(); X=data.data;y=data.target
    rows=[]
    for seed in seeds:
        split=split_data(y,seed)
        models={
            "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200)),
            "random_forest":RandomForestClassifier(n_estimators=150,min_samples_leaf=2,random_state=seed,n_jobs=1)
        }
        for name,model in models.items():
            model.fit(X[split.train],y[split.train])
            t=fit_temperature(model.predict_proba(X[split.temperature]),y[split.temperature])
            calibrated=apply_temperature(model.predict_proba(X[split.conformal]),t)
            cutoff=conformal_threshold(calibrated,y[split.conformal])
            for sigma in sigmas:
                x=corrupt(X[split.test],sigma,seed*10000+int(sigma*100))
                raw=model.predict_proba(x)
                adjusted=apply_temperature(raw,t)
                sets=prediction_sets(adjusted,cutoff)
                cov=float(np.mean(sets[np.arange(len(split.test)),y[split.test]]))
                size=float(sets.sum(axis=1).mean())
                for method,p in (("raw",raw),("temperature_scaled",adjusted)):
                    rows.append(dict(seed=seed,model=name,sigma=sigma,method=method,
                        n_test=len(split.test),temperature=t,conformal_coverage=cov,conformal_set_size=size,
                        **evaluate(p,y[split.test])))
    return rows
