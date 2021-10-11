import time
from sklearn.preprocessing import RobustScaler, StandardScaler
import sklearn.metrics as mt
from pandas import DataFrame

named_scalers = {}


def df_fit(model, df, *a, **k):
    """
    sklearn fit wrapper, using the column named `y` in
    df as the label.
    """
    x = df.drop(["y"], axis=1)
    y = df["y"]
    return model.fit(x, y, *a, **k)


def df_predict(model, df, *a, **k):
    """
    sklearn fit wrapper, using the column named `y` in
    df as the label.
    """
    x = df[[c for c in df if c != "y"]]
    return model.predict(x, *a, **k)


def df_predict_proba(model, df, *a, **k):
    """
    sklearn fit wrapper, using the column named `y` in
    df as the label.
    """
    x = df[[c for c in df if c != "y"]]
    return model.predict_proba(x, *a, **k)


def set_df_funcs(sklearn):
    sklearn.base.ClassifierMixin.dfit = df_fit
    sklearn.base.ClassifierMixin.dpredict = df_predict
    sklearn.base.ClassifierMixin.dpredict_proba = df_predict_proba


def std_sc(srs, name=None, rob=False, mk_sc=None):
    if mk_sc is not None:
        ss = mk_sc()
    else:
        ss = RobustScaler() if rob else StandardScaler()
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


def shapes(*xs):
    return [x.shape for x in xs]
