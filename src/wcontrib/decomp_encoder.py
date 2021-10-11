"""Coocurrence Encoder"""
from collections import defaultdict
from itertools import count

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

# from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = "wcbeard"


class CoocurrenceEncoder(BaseEstimator, TransformerMixin):
    """<> encoding for categorical features.

    <Description>

    <Parameters>
    ----------
    cols : tuple, list of tuples

    decomp_model :

    tfidf : whether to TF-IDF transform the co-occurrence count matrix
    before matrix decomposition.

    """

    def __init__(
        self,
        cols,
        drop_original_cols=True,
        decomp_model="nmf",
        n_components=5,
        # feature_prefix='n',
        log_cts=True,
        tfidf=False,
        verbose=0,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        # min_samples_leaf=1,
        # smoothing=1.0,
    ):
        # assert isinstance(
        #     cols, (tuple, list)
        # ), "cols must be a tuple or list of tuples"
        check_pair_or_list_of_pairs(cols)
        if isinstance(cols[0], (tuple, list)):
            self.col_pairs_list = cols
        else:
            self.col_pairs_list = [cols]
        # self.cols = cols
        self.drop_original_cols = drop_original_cols

        if isinstance(decomp_model, str):
            if decomp_model == "nmf":
                self.decomp_model = NMF(n_components=n_components)
            elif decomp_model == "svd":
                self.decomp_model = TruncatedSVD(
                    n_components=n_components
                )
            else:
                raise ValueError(
                    "`decomp_model` should be a matrix factorization estimator, "
                    "or string 'nmf' or 'svd'. {} was passed".format(
                        decomp_model
                    )
                )
        else:
            self.decomp_model = decomp_model

        self.n_components = n_components
        # if feature_prefix is not None:
        #     self.feature_prefix = feature_prefix
        # else:
        #     model_name = type(self.decomp_model).__name__.lower()
        #     self.feature_prefix = model_name[0]

        self.log_cts = log_cts
        self.tfidf = tfidf
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        # TODO: option to drop originals
        self.drop_cols = []
        self.verbose = verbose
        # self.ordinal_encoder = None
        # self.min_samples_leaf = min_samples_leaf
        # self.smoothing = float(
        #     smoothing
        # )
        self._dim = None
        self.embeddings = {}  # √
        # TODO: do I use handle_unknown?
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.feature_names = None

    @classmethod
    def _coocurrence_matrix_from_pairs(
        self, pairs, weights=None, tfidf=False
    ):
        """
        Parameters
        ----------



        TODO: iterable

        pairs : array-like, shape = [n_samples, 2]

        Create dict of dicts where keys are mapped from a/b
        A will be keys/columns
        B will be row indices
        """
        # if not isinstance(pairs, list):
        #     pairs = list(pairs)

        if weights is None:
            weights = np.ones(len(pairs))

        # Create dict of dicts where keys are original a/b values
        a_to_b_dicts = defaultdict(lambda: defaultdict(int))
        for (a, b), w in zip(pairs, weights):
            a_to_b_dicts[a][b] += w

        sdf = self._cooc_dict_to_sparse_df(a_to_b_dicts, tfidf=tfidf)
        return sdf

    @staticmethod
    def _cooc_dict_to_sparse_df(a_to_b_dicts, tfidf=False):
        """


        This method converts the dict of dicts into a sparse dataframe, where the
        columns consists of the unique a elements, the index consists of the
        unique b elements, and the entries of the dataframe are the co-occurring counts.


        a_to_b_dicts: a co-ocurrence dict of dicts. This dict is generated from
        pairs (a, b) that occur together and has the type
        Dict[a, Dict[b, (a, b co-occurrence count)]].

        The keys in `a_to_b_dicts` are python objects, and some indexing is done
        to convert them to integer
        ....
        """
        a_vals = list(a_to_b_dicts)
        b_vals = {
            b
            for a, b_dict in a_to_b_dicts.items()
            for b in b_dict.keys()
        }
        a_index, b_index = map(sorted_set, [a_vals, b_vals])
        a2ix = dict(zip(a_index, count()))
        b2ix = dict(zip(b_index, count()))
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

        if tfidf:
            tfidf_mtx = TfidfTransformer().fit_transform(
                sdf.sparse.to_coo()
            )
            sdf_tfidf = pd.DataFrame.sparse.from_spmatrix(
                tfidf_mtx, index=sdf.index, columns=sdf.columns
            )
            return sdf_tfidf

        return sdf

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : encoder
            Returns self.

        """

        # unite the input into pandas types
        X = util.convert_input(X)
        # y = util.convert_input_vector(y, X.index)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of X is {} but length of y is {}.".format(
                    X.shape[0], y.shape[0]
                )
            )

        self._dim = X.shape[1]

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError(
                    "Columns to be encoded can not contain null"
                )

        self._fit_cooc_encode(self, X, y=None)

        # TODO: after transform
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(
                X, X_temp, self.cols
            )
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5
            ]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print(
                        "Could not remove column from feature names."
                        "Not found in generated cols.\n{}".format(e)
                    )

        return self

    def _fit_cooc_encode(self, X_in, y=None):
        """Perform the co-occurrence encoding."""
        X = X_in.copy(deep=True)
        original_feature_names = set()

        for cols_pair in self.col_pairs_list:
            original_feature_names.update(set(cols_pair))
            self._fit_cooc_encode_single(X, cols_pair)
        original_feature_names = sorted(original_feature_names)
        if self.drop_original_cols:
            self.drop_cols.extend(original_feature_names)

        self.feature_names = [
            new_feature_name
            for (
                feature_a,
                feature_b,
            ), embedding in self.embeddings.items()
            for new_feature_name in embedding.columns
        ]

        return self

    def _transform_cooc_encode(self, X_in, y=None):
        X = X_in.copy(deep=True)
        new_embs = []
        for (
            feature_a,
            feature_b,
        ), embedding in self.embeddings.items():
            reshaped_embedding = embedding.reindex(X[feature_a])
            reshaped_embedding.index = X.index
            new_embs.append(reshaped_embedding)

        new_feature_df = pd.concat(
            new_embs, ignore_index=False, axis=1
        )
        for new_feature_col in new_feature_df:
            X[new_feature_col] = new_feature_df[new_feature_col]
        if self.drop_cols:
            X = X.drop(self.drop_cols, axis=1)
        return X

    def _fit_cooc_encode_single(self, X, cols_pair):
        # QA: review this
        feature_a, feature_b = cols_pair = list(cols_pair)
        a_b_pairs = X[cols_pair].values
        cooc_sdf = self._coocurrence_matrix_from_pairs(
            a_b_pairs, weights=None, tfidf=self.tfidf
        )
        if self.log_cts:
            cooc_sdf = np.log10(cooc_sdf + 1)

        cooc_coo = cooc_sdf.sparse.to_coo()

        self.decomp_model.fit(cooc_coo)
        emb_a = pd.DataFrame(
            self.decomp_model.components_.T,
            index=cooc_sdf.columns,
            columns=[
                "{}__{}{}".format(feature_a, feature_b, i)
                for i in range(1, self.n_components + 1)
            ],
        )
        emb_a.index.name = feature_a
        emb_a.columns.name = feature_b

        emb_b = pd.DataFrame(
            self.decomp_model.transform(cooc_coo),
            index=cooc_sdf.index,
            columns=[
                "{}__{}{}".format(feature_b, feature_a, i)
                for i in range(1, self.n_components + 1)
            ],
        )
        emb_b.index.name = feature_b
        emb_b.columns.name = feature_a
        self.embeddings[(feature_a, feature_b)] = emb_a
        self.embeddings[(feature_b, feature_a)] = emb_b

    def fit_target_encoding(self, X, y):
        mapping = {}

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            values = switch.get("mapping")

            prior = self._mean = y.mean()

            stats = y.groupby(X[col]).agg(["count", "mean"])

            smoove = 1 / (
                1
                + np.exp(
                    -(stats["count"] - self.min_samples_leaf)
                    / self.smoothing
                )
            )
            smoothing = prior * (1 - smoove) + stats["mean"] * smoove
            smoothing[stats["count"] == 1] = prior

            if self.handle_unknown == "return_nan":
                smoothing.loc[-1] = np.nan
            elif self.handle_unknown == "value":
                smoothing.loc[-1] = prior

            if self.handle_missing == "return_nan":
                smoothing.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == "value":
                smoothing.loc[-2] = prior

            mapping[col] = smoothing

        return mapping

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target info (such as transform test set)

        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError(
                    "Columns to be encoded can not contain null"
                )

        if self._dim is None:
            raise ValueError(
                "Must train encoder before it can be used to transform data."
            )

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError(
                "Unexpected input dimension %d, expected %d"
                % (
                    X.shape[1],
                    self._dim,
                )
            )

        if not list(self.cols):
            return X

        X = self._transform_cooc_encode(X)

        # TODO:
        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError(
                    "Unexpected categories found in dataframe"
                )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        # TODO:
        if not isinstance(self.feature_names, list):
            raise ValueError(
                "Must fit data first. Affected feature names are not known before."
            )
        else:
            return self.feature_names


def sorted_set(xs):
    return sorted(set(xs))


def check_pair_or_list_of_pairs(xs):
    def check_pair(tup):
        if len(tup) != 2:
            raise ValueError("Must be a pair or list of pairs.")

    if not isinstance(xs, (tuple, list)):
        raise ValueError("Must be a pair or list of pairs.")
    if len(xs) == 0:
        raise ValueError(
            "Tuple of length 2, or list of tuples expected. Got an empty container."
        )
    is_pair = not isinstance(xs[0], (tuple, list))
    if is_pair:
        check_pair(xs)
    else:
        for x in xs:
            check_pair(x)


def test_check_pair_or_list_of_pairs():
    from pytest import raises

    check_pair_or_list_of_pairs(("a", "b"))
    check_pair_or_list_of_pairs([("a", "b")])
    check_pair_or_list_of_pairs([("a", "b"), ("c", "d")])

    with raises(ValueError):
        check_pair_or_list_of_pairs(1)
    with raises(ValueError):
        check_pair_or_list_of_pairs([1])
    with raises(ValueError):
        check_pair_or_list_of_pairs(([1], [2]))
    with raises(ValueError):
        check_pair_or_list_of_pairs([])


xsamp = [
    ("a", "b", 1),
    ("a", "b", 1),
    ("a", "d", 1),
    ("z", "d", 1),
    ("c", "b", 1),
]
xsamp_df = pd.DataFrame(xsamp, columns=["aa", "bb", "cc"])


def test_pairs_to_cooc_sparse_df():
    from pandas.testing import assert_frame_equal

    xsamp2 = [(a, b) for a, b, c in xsamp]
    sdf = CoocurrenceEncoder._coocurrence_matrix_from_pairs(xsamp2)

    expected_df = pd.DataFrame(
        np.matrix("2 1 0;1 0 1"),
        columns=["a", "c", "z"],
        index=["b", "d"],
    )
    assert_frame_equal(sdf.sparse.to_dense(), expected_df)
    assert (sdf.dtypes == "Sparse[int64, 0]").all()

    sdf_tfidf = CoocurrenceEncoder._coocurrence_matrix_from_pairs(
        xsamp2, tfidf=True
    )
    expected_tfidf = pd.DataFrame(
        np.matrix("0.82 0.57 0; 0.58 0 0.81"),
        index=sdf_tfidf.index,
        columns=sdf_tfidf.columns,
    )
    assert_frame_equal(
        sdf_tfidf.sparse.to_dense().round(2), expected_tfidf
    )


def test_fit_cooc_encode():
    coe = CoocurrenceEncoder(["aa", "bb"])
    coe = coe._fit_cooc_encode(xsamp_df)
    assert coe.feature_names == [
        "aa__bb1",
        "aa__bb2",
        "aa__bb3",
        "aa__bb4",
        "aa__bb5",
        "bb__aa1",
        "bb__aa2",
        "bb__aa3",
        "bb__aa4",
        "bb__aa5",
    ]
    assert len(coe.embeddings) == 2


def test_fit_cooc_encode_single():
    coe = CoocurrenceEncoder(["aa", "bb"], n_components=5)
    coe._fit_cooc_encode_single(xsamp_df, ["aa", "bb"])

    emb_a = coe.embeddings["aa", "bb"]
    assert (
        emb_a.columns
        == ["aa__bb1", "aa__bb2", "aa__bb3", "aa__bb4", "aa__bb5"]
    ).all()
    assert (emb_a.index == sorted_set(xsamp_df["aa"])).all()

    emb_b = coe.embeddings["bb", "aa"]
    assert (
        emb_b.columns
        == ["bb__aa1", "bb__aa2", "bb__aa3", "bb__aa4", "bb__aa5"]
    ).all()
    assert (emb_b.index == sorted_set(xsamp_df["bb"])).all()


def test_transform_cooc_encode():
    coe = CoocurrenceEncoder(["aa", "bb"])._fit_cooc_encode(xsamp_df)
    X2 = coe._transform_cooc_encode(xsamp_df)
    assert (
        X2.columns
        == [
            "cc",
            "aa__bb1",
            "aa__bb2",
            "aa__bb3",
            "aa__bb4",
            "aa__bb5",
            "bb__aa1",
            "bb__aa2",
            "bb__aa3",
            "bb__aa4",
            "bb__aa5",
        ]
    ).all()
    assert (xsamp_df.index == X2.index).all()
    assert len(xsamp_df) == len(X2)
