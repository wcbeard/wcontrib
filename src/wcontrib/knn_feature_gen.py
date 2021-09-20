from enum import Flag, auto
from functools import reduce, wraps

from numba import njit
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors


def simple_trace(f):
    @wraps(f)
    def wrapper(*a, **k):
        try:
            return f(*a, **k)
        except Exception as e:
            print(f"{type(e).__name__}:")
            print(e)

    return wrapper


class KnnOpts(Flag):
    frac_x_per_class = auto()
    max_k_streak = auto()
    dist_to_class = auto()
    norm_dist_to_class = auto()
    dist_to_kth_neighbor = auto()
    mean_dist_to_neighbor = auto()


default_opts = reduce(lambda a, b: a | b, KnnOpts)


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_jobs,
        k_list,
        metric,
        n_classes=None,
        n_neighbors=None,
        eps=1e-6,
        opts=default_opts,
    ):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric

        if n_neighbors is None:
            self.n_neighbors = max(k_list)
        else:
            self.n_neighbors = n_neighbors

        self.eps = eps
        self.opts = opts
        self.n_classes_ = n_classes

    def fit(self, X, y):
        """
        Set's up the train set and self.NN object
        """
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function
        self.NN = NearestNeighbors(
            n_neighbors=max(self.k_list),
            metric=self.metric,
            n_jobs=1,
            algorithm="brute" if self.metric == "cosine" else "auto",
        )
        self.NN.fit(X)

        # Store labels
        self.y_train = y

        # Save how many classes we have
        self.n_classes = (
            len(np.unique(y)) if self.n_classes_ is None else self.n_classes_
        )

    def predict(self, x):
        """
        Computes KNN features for a single object `x`
        """
        N = x.shape[0]
        kdist, kind = self.NN.kneighbors(x)
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[kind]

        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = []

        if KnnOpts.frac_x_per_class in self.opts:
            for k in self.k_list:
                feats = frac_objects_each_class(k, neighs_y, n_classes=self.n_classes)
                assert (
                    len(feats.T) == self.n_classes
                ), f"{len(feats)} ≠ {self.n_classes}"
                return_list += [feats]

        """
        Same label streak: the largest number N, 
        such that N nearest neighbors have the same label.
        """
        if KnnOpts.max_k_streak in self.opts:
            feats_streak = max_k_streak(neighs_y)
            assert len(feats_streak) == N
            return_list += [feats_streak]

        """
        Minimum distance to objects of each class
        Find the first instance of a class and take its distance as features.
        """
        # This will be a list of arrays, where each array matches
        # the dimension of the labels y (so N).
        # To make it compatible with the other features being
        # packed into `return_list`, stack into a rectangular matrix
        # and transpose.
        if KnnOpts.dist_to_class in self.opts:
            feats_min_dist = []
            for c in range(self.n_classes):
                _feats_c = dist_to_class(c, kdist, neighs_y, default_dist=999)
                feats_min_dist.append(_feats_c)

            assert len(feats_min_dist) == self.n_classes
            feats_arr = col_list_to_arr(feats_min_dist)
            # `feats_arr` should be reshaped so its 1st column is the same
            # as the first array in `feats_min_dist`
            assert (feats_min_dist[0] == feats_arr[:, 0]).all()
            return_list += [feats_arr]

        """
        Minimum *normalized* distance to objects of each class
        As 3. but we normalize (divide) the distances
        by the distance to the closest neighbor.

        If there are no neighboring objects of some classes, 
        Then set distance to that class to be 999.
        """
        if KnnOpts.norm_dist_to_class in self.opts:
            feats = []
            for c in range(self.n_classes):
                _feats_c = dist_to_class(c, kdist, neighs_y, default_dist=999)
                _feats_c_normed = _feats_c / (kdist.min(axis=1) + self.eps)
                feats.append(_feats_c_normed)

            assert len(feats) == self.n_classes
            feats_arr = col_list_to_arr(feats)
            # `feats_arr` should be reshaped so its 1st column is the same
            # as the first array in `feats`
            assert (feats[0] == feats_arr[:, 0]).all()
            return_list += [feats_arr]

        """
        5.1 Distance to Kth neighbor
           Think of this as of quantiles of a distribution
        5.2 Distance to Kth neighbor normalized by 
           distance to the first neighbor
        """
        if KnnOpts.dist_to_kth_neighbor in self.opts:

            feat_51 = []
            feat_52 = []

            for k in self.k_list:
                feat_51_k = kdist[:, k - 1]
                feat_52_k = kdist[:, k - 1] / (kdist[:, 0] + self.eps)
                feat_51.append(feat_51_k)
                feat_52.append(feat_52_k)

            feat_51 = col_list_to_arr(feat_51)
            feat_52 = col_list_to_arr(feat_52)
            return_list += [feat_51, feat_52]

        """
        Mean distance to neighbors of each class for each K from `k_list` 
        For each class select the neighbors of that class among K nearest neighbors 
        and compute the average distance to those objects

        If there are no objects of a certain class among K neighbors,
        set mean distance to 999

        You can use `np.bincount` with appropriate weights
        Don't forget, that if you divide by something, 
        You need to add `self.eps` to denominator.
        """
        if KnnOpts.mean_dist_to_neighbor in self.opts:
            for k in self.k_list:
                feats_k = mean_dist_to_class(k, neighs_y, kdist, self.n_classes_)
                # YOUR CODE GOES IN HERE
                assert feats_k.shape == (N, self.n_classes)
                # assert len(feats_k) == self.n_classes
                return_list += [feats_k]

        return np.column_stack(return_list)


def frac_objects_each_class(k, neighs_y, n_classes):
    """
    Fraction of objects of every class.
    It is basically a KNNСlassifiers predictions.

    Take a look at `np.bincount` function, it can be very helpful
    Note that the values should sum up to one
    """
    neighs_y_k = neighs_y[:, :k]
    feats = np.vstack([np.bincount(y_row, minlength=n_classes) for y_row in neighs_y_k])
    feats_normed = np.divide(feats, feats.sum(axis=1)[:, None])
    return feats_normed


def all_same_in_row(X):
    fst_col = X[:, [0]]
    eq_x = np.equal(X, fst_col)
    all_eq = np.all(eq_x, axis=1)
    return all_eq


def test_all_same_in_row():
    res = all_same_in_row(np.array([[1, 2, 3], [0, 0, 0]]))
    assert (res == [False, True]).all()


@njit
def max_k_streak(neighs_y):
    """
    neighs_y: [[ 1, 1, 13],
               [15,  4, 19]]
    means that 2 nearest neighbors of 1st observation
    have label 1. So for that obs, the resulting output
    should be 2. For the 2nd obs, the output should be 1.
    """
    n = len(neighs_y)
    res = np.zeros(n)
    for i in range(n):
        streak = 0
        row = neighs_y[i]
        for e in row:
            if e != row[0]:
                break
            streak += 1
        res[i] = streak

    return res


def dist_to_class(cls, kdist, neighs_y, default_dist=999):
    """
    n: # observations
    k: k nearest neighbors

    cls: int (class label)
    kdist ∈ R^{n x k}
    neighs_y ∈ cls^{n x k}
    """
    cls_bm = neighs_y == cls
    # off-class distances set to default_dist
    class_dist = np.where(cls_bm, kdist, default_dist)
    closest_cls_dist = class_dist.min(axis=1)
    return closest_cls_dist


@simple_trace
@njit
def mean_dist_to_class(k, neighs_y, kdist, n_classes, default_dist=999):
    N = len(neighs_y)
    res = np.zeros((N, n_classes))
    for c in range(n_classes):
        for n in range(N):
            row_labels = neighs_y[n]
            class_bm = row_labels == c
            dists = kdist[n][class_bm]
            if not len(dists):
                res[n, c] = default_dist
            else:
                res[n, c] = np.mean(dists)
    return res


def col_list_to_arr(aa):
    return np.vstack(aa).T


def example_usage(X, Y, X_test):
    xt = X[:10]
    y = Y[:10]
    xv = X_test[:10]
    k_list = [3, 8, 32]

    NNF = NearestNeighborsFeats(
        n_jobs=1, k_list=k_list, metric="minkowski", n_classes=len(np.unique(Y))
    )
    NNF.fit(xt, y)
    res = NNF.get_features(xv)
    return res


def test_max_k_streak():
    a_1_2_3 = np.matrix("1 21 13; 15 15 19; 4 4 4", dtype=int)
    assert (max_k_streak(a_1_2_3) == [1, 2, 3]).all()


def test_dist_to_class():
    kdist = np.matrix("1 2 3; 4 5 6; 7.9 9 10")
    neighbors_y = np.matrix("0 1 2; 1 2 0; 0 0 0")
    assert (dist_to_class(0, kdist, neighbors_y) == np.array([1.0, 6.0, 7.9])).all()
    assert (dist_to_class(1, kdist, neighbors_y) == np.array([2.0, 4.0, 999.0])).all()


def test_col_list_to_arr():
    ab = [np.array([1, 2, 3]), np.array([3, 4, 5])]
    res = col_list_to_arr(ab)
    assert (res == np.matrix("1 3; 2 4; 3 5")).all()


def run_tests():
    test_col_list_to_arr()
    test_dist_to_class()
    test_all_same_in_row()
