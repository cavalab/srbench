#!/usr/bin/env python

"""Tests for `pstree` package."""

from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor


def test_simple_data():
    X, y = make_regression(n_samples=100, n_features=5, n_informative=5)
    gp = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                         height_limit=6, n_pop=20, n_gen=2,
                         min_samples_leaf=1, max_leaf_nodes=None,
                         adaptive_tree=False, basic_primitive='optimal', size_objective=True)
    gp.fit(X, y)
    assert_almost_equal(mean_squared_error(y, gp.predict(X)), 0)


def test_adaptive_tree():
    X, y = make_regression(n_samples=100, n_features=5, n_informative=5)
    gp = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                         height_limit=6, n_pop=20, n_gen=5,
                         adaptive_tree=True, basic_primitive='optimal', size_objective=True)
    gp.fit(X, y)
    assert r2_score(y, gp.predict(X)) > 0
