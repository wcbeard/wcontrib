import time
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as mt
from pandas import DataFrame

named_scalers = {}


def std_sc(srs, name=None):
    ss = StandardScaler()
    if name is not None:
        named_scalers[name] = ss
    return ss.fit_transform(srs.values[:, None]).ravel()


def coefs(coefs, names=None):
    if names is None:
        names = range(len(coefs))
    df = (
        DataFrame({"c": coefs, "n": names})
        .assign(ca=lambda df: df.c.abs())
        .sort_values("ca", ascending=False)
        .drop(["ca"], axis=1)
    )
    return df


def auc_pr(yt, pred):
    prec, rec, thresh = mt.precision_recall_curve(yt, pred)
    return mt.auc(rec, prec)


def tic():
    tic.start = time.perf_counter()


def toc():
    end = time.perf_counter()
    return end - tic.start

