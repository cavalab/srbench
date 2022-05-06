import numpy as np
from sklearn import clone
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils import safe_sqr

from .gp_function import max, min


class LassoRidge():
    def __init__(self, lasso_model, ridge_model, *args, **kwargs):
        """
        Using Lasso to select useful variables, and using Ridge to fit the final model
        """
        super().__init__(*args, **kwargs)
        self.lasso_model = lasso_model
        self.ridge_model = ridge_model

    def fit(self, X, y, sample_weight=None):
        pipe = Pipeline(
            [("Selector", SelectFromModel(self.lasso_model, threshold=1e-20)),
             ("Ridge", self.ridge_model)]
        )
        # pipe.fit(X, y, Selector__sample_weight=sample_weight, Ridge__sample_weight=sample_weight)
        pipe.fit(X, y, Ridge__sample_weight=sample_weight)

        feature = np.zeros(X.shape[1])
        feature[pipe['Selector'].estimator_.coef_ != 0] = pipe['Ridge'].coef_
        self.coef_ = feature
        self.intercept_ = pipe['Ridge'].intercept_
        self.pipe = pipe
        return pipe

    def predict(self, X):
        return self.pipe.predict(X)


class RFERegressor(RFE):
    def fit(self, X, y, sample_weight=None):
        self.sample_weight = sample_weight
        super().fit(X, y)
        self.coef_ = np.array(self.support_).astype(float)
        self.intercept_ = self.estimator_.intercept_

    def _fit(self, X, y, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        tags = self._get_tags()
        X, y = self._validate_data(
            X, y, accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get('allow_nan', True),
            multi_output=True
        )
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y, sample_weight=self.sample_weight)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)
            # Eliminate useless features (ensure at least one feature is retained)
            threshold = max(threshold, min(np.count_nonzero(coefs == 0), len(coefs) - 1))

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self
