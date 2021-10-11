from collections import Counter
from functools import reduce

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin


def str_summ(xs, n=2):
    ctr = Counter(xs)
    mc = ctr.most_common(n)
    lst_vc = [f"{val}:{ct}" for val, ct in mc]
    if len(ctr) > n:
        sum_so_far = sum(c for v, c in mc)
        sum_left = len(xs) - sum_so_far
        lst_vc.append(f"xx: {int(sum_left / len(xs) * 100)}")
    return "  ".join(lst_vc)


def reduce_join(df, columns):
    assert len(columns) > 1
    slist = [df[x].astype(str) for x in columns]
    return reduce(lambda x, y: x + "_" + y, slist[1:], slist[0])


def process_df(df, id_fields):
    df = df.rename(columns=str.lower)
    cat_fields = reduce_join(df, id_fields)
    df["ids"] = LabelEncoder().fit_transform(cat_fields)
    df["id_card"] = df.groupby("ids").action.transform("size")
    # role_title is 100% correlated with role_code
    df = df.rename(columns={"action": "y"}).drop(
        ["role_title"], axis=1
    )

    dft, dfv = train_test_split(
        df,
        test_size=None,
        train_size=0.9,
        random_state=0,
        shuffle=True,
        stratify=None,
    )
    return df, dft, dfv


#################
# Mean encoding #
#################
def mean_encoding_c(pc, nc, p_global, a=6):
    num = pc * nc + p_global * a
    return num / (nc + a)


class MeanEnc(TransformerMixin):
    def __init__(self, cols, alpha=6):
        self.alpha = alpha
        self.cols = cols
        self.col_mapping = None

    def fit(self, df, y: str = "y"):
        assert isinstance(y, str), "y must be column name"
        self.col_mapping = {}

        for col in self.cols:
            agg = df.groupby([col])[y].agg(["mean", "size"])
            enc_val_agg = mean_encoding_c(
                agg["mean"],
                agg["size"],
                agg["mean"].mean(),
                a=self.alpha,
            )
            self.col_mapping[col] = enc_val_agg.to_dict()
        return self

    def transform(self, df):
        df = df.copy()
        for col, mapping in self.col_mapping.items():
            col_mean = np.mean(list(mapping.values()))
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(col_mean)
        return df


def _mean_encoding(df, cat, y, a=6):
    agg = df.groupby([cat])[y].agg(["mean", "size"])
    enc_val_agg = mean_encoding_c(
        agg["mean"], agg["size"], agg["mean"].mean(), a=a
    )
    return df[cat].map(enc_val_agg.to_dict())


#######################
# Variable embeddings #
#######################
from sklearn.decomposition import NMF  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa
import wcontrib.cooc as wcooc  # noqa

conf = lambda: None
conf.df = None


def var_embedding(
    a, b, df=None, n_components=2, ret_b_emb=False, log_cts=False
):
    """
    Get embeddings of a wrt b.
    """
    if df is None:
        df = conf.df
    sdf = wcooc.pairs_to_cooc_sparse_df(
        df[[a, b]].itertuples(index=False, name=None)
    )
    if log_cts:
        sdf = np.log10(sdf + 1)
    X = sdf.sparse.to_coo()

    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    emb_a = pd.DataFrame(
        nmf.components_.T,
        index=sdf.columns,
        columns=[f"n{i}" for i in range(1, n_components + 1)],
    )
    emb_a.index.name = a
    emb_a.columns.name = b

    if ret_b_emb:
        emb_b = pd.DataFrame(
            nmf.fit_transform(X),
            index=sdf.index,
            columns=[f"n{i}" for i in range(1, n_components + 1)],
        )
        emb_b.index.name = b
        emb_b.columns.name = a
        return emb_a, emb_b
    return emb_a
