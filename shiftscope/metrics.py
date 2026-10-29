"""Calibration, uncertainty, and conformal scores for multiclass classifiers."""
import numpy as np
from scipy.optimize import minimize_scalar

def validated(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError("Expected finite normalized class probabilities")
    return p

def apply_temperature(p,t):
    p=validated(p)
    if not np.isfinite(t) or t<=0: raise ValueError("Temperature must be positive")
    z=np.log(np.clip(p,1e-12,1))/t
    z-=z.max(axis=1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(axis=1,keepdims=True)

def nll(p,y):
    p=validated(p); y=np.asarray(y,dtype=int)
    return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())

def fit_temperature(p,y):
    result=minimize_scalar(lambda x:nll(apply_temperature(p,np.exp(x)),y),bounds=(np.log(.3),np.log(8)),method="bounded")
    if not result.success: raise RuntimeError("Temperature fit failed")
    return float(np.exp(result.x))

def ece(p,y,bins=10):
    p=validated(p)
    conf=p.max(axis=1); ok=(p.argmax(axis=1)==y).astype(float)
    indexes=np.minimum((conf*bins).astype(int),bins-1)
    return float(sum((indexes==k).mean()*abs(ok[indexes==k].mean()-conf[indexes==k].mean()) for k in range(bins) if (indexes==k).any()))

def brier(p,y):
    p=validated(p)
    return float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)))

def aurc(p,y):
    p=validated(p); order=np.argsort(-p.max(axis=1),kind="stable")
    err=(p.argmax(axis=1)[order]!=y).astype(float)
    return float(np.mean(np.cumsum(err)/np.arange(1,len(y)+1)))

def conformal_threshold(p,y,alpha=.1):
    p=validated(p)
    if not 0<alpha<1: raise ValueError("alpha must be in (0,1)")
    scores=np.sort(1-p[np.arange(len(y)),y])
    k=int(np.ceil((len(y)+1)*(1-alpha)))
    return float(scores[k-1]) if k<=len(scores) else 1.

def prediction_sets(p,threshold):
    return (1-validated(p))<=threshold

def evaluate(p,y):
    p=validated(p)
    return dict(accuracy=float(np.mean(p.argmax(axis=1)==y)),ece=ece(p,y),
        nll=nll(p,y),brier=brier(p,y),aurc=aurc(p,y),
        mean_confidence=float(p.max(axis=1).mean()))
