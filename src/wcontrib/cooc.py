from collections import defaultdict
import itertools as it

import numpy as np
from pandas.testing import assert_frame_equal
import pandas as pd
import scipy.sparse as sp


def pairs_to_cooc_sparse_df(ab_pairs, weights=None):
    """"""
    ab_pairs = (
        ab_pairs if isinstance(ab_pairs, list) else list(ab_pairs)
    )

    if weights is None:
        weights = np.ones(len(ab_pairs))

    # Create dict of dicts where keys are original a/b values
    a_to_b_dicts = defaultdict(lambda: defaultdict(int))
    for (a, b), w in zip(ab_pairs, weights):
        a_to_b_dicts[a][b] += w

    sdf = cooc_df_to_sparse_df(a_to_b_dicts)
    return sdf


def cooc_df_to_sparse_df(a_to_b_dicts):
    """
    Create dict of dicts where keys are mapped from a/b
    A will be keys/columns
    B will be row indices
    """
    a_vals = list(a_to_b_dicts)
    b_vals = {
        b
        for a, b_dict in a_to_b_dicts.items()
        for b, _ in b_dict.items()
    }
    a_index, b_index = map(sorted_set, [a_vals, b_vals])
    a2ix = dict(zip(a_index, it.count()))
    b2ix = dict(zip(b_index, it.count()))
    A = len(a2ix)
    B = len(b2ix)

    mtx = sp.dok_matrix((B, A), dtype=np.int64)
    for a, b_dict in sorted(a_to_b_dicts.items()):
        ix_a = a2ix[a]
        for b, cnt in b_dict.items():
            ix_b = b2ix[b]
            mtx[ix_b, ix_a] = cnt

    sdf = pd.DataFrame.sparse.from_spmatrix(
        mtx, index=b_index, columns=a_index
    )
    return sdf


def test_pairs_to_cooc_sparse_df():
    xsamp = [("a", "b"), ("a", "d"), ("z", "d"), ("c", "b")]
    sdf = pairs_to_cooc_sparse_df(xsamp)

    res = pd.DataFrame(
        np.matrix("1 1 0;1 0 1"),
        columns=["a", "c", "z"],
        index=["b", "d"],
    )
    assert_frame_equal(sdf.sparse.to_dense(), res)
    assert (sdf.dtypes == "Sparse[int64, 0]").all()


def cooc_to_sparse_df_slow(ab_pairs, weights=None):
    """
    Sparse DataFrame's are nice because they automatically
    do the column and index book-keeping. Unfortunately, the
    most intuitive way I can find to create them is to first
    densify each series, and turn it into a Sparse array
    before creating a sparse df out of it. Hopefully there
    won't be memory issues from expanding it.

    Follow up: this is slow, but could be a useful reference
    implementation for testing.
    """
    ab_pairs = (
        ab_pairs if isinstance(ab_pairs, list) else list(ab_pairs)
    )
    if weights is None:
        weights = np.ones(len(ab_pairs))
    a_to_b_dicts = defaultdict(lambda: defaultdict(int))
    for (a, b), w in zip(ab_pairs, weights):
        a_to_b_dicts[a][b] += w

    # Densify

    b_index = sorted(set(b for a, b in ab_pairs))
    a_to_sparse_b = {}
    for a, b_dict in sorted(a_to_b_dicts.items()):
        dense_b_srs = pd.Series(
            [b_dict.get(i, np.nan) for i in b_index], index=b_index
        )
        sparse_b_srs = pd.arrays.SparseArray(dense_b_srs)
        a_to_sparse_b[a] = sparse_b_srs

    sdf = pd.DataFrame(a_to_sparse_b).fillna(0).astype(int)
    sdf.index = b_index
    return sdf


def concat_embeddings(df, emb, pref="u"):
    cols = [f"{pref}{i}" for i in range(1, emb.shape[1] + 1)]
    kwa = {c: emb[:, i] for i, c in enumerate(cols)}
    return df.assign(**kwa)


"""
from sklearn.feature_extraction.text import CountVectorizer

def pairs_to_cooc_df(
    X, diag_val=None, norm_diag=False, binary_cooc=False
):
    count_model = CountVectorizer(
        ngram_range=(1, 1), analyzer=lambda x: x
    ).fit(X)
    xx = count_model.transform(X)

    if binary_cooc:
        xx[xx > 0] = 1
    Xc = xx.T * xx

    if diag_val is not None:
        Xc.setdiag(diag_val)

    if norm_diag:
        g = sp.diags(1.0 / Xc.diagonal())
        Xc = g * Xc

    sdf = pd.DataFrame.sparse.from_spmatrix(
        Xc,
        index=count_model.get_feature_names(),
        columns=count_model.get_feature_names(),
    )
    return sdf


def cooc_agg(pairs, weights=None):
    if weights is None:
        weights = np.ones(len(pairs))
    dct = defaultdict(int)
    for e1, e2 in pairs:
        # zip(x1, x2):
        dct[(e1, e2)] += 1
    return dct


"""
