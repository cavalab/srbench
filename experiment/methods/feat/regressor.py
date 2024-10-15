# This example submission shows the submission of FEAT (cavalab.org/feat). 
from feat import FeatRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from feat import Feat, FeatRegressor, FeatClassifier 

from sklearn.datasets import load_diabetes, make_blobs
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import unittest
import argparse
import sys
import pandas as pd
import numpy as np
import pickle


"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
est:RegressorMixin = FeatRegressor(
                    pop_size=100,
                    gens=100,
                    max_time=8*60*60,  # 8 hrs. Your algorithm should have this feature
                    max_depth=6,
                    verbosity=2,
                    batch_size=100,
                    functions=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log'],
                    otype='f'
                   )
# want to tune your estimator? wrap it in a sklearn CV class. 


class FeatPopEstimator(RegressorMixin):
    """
    FeatPopEstimator is a custom regressor that wraps a fitted FEAT estimator
    to call `model` and `predict` from its archive.
    
    Attributes:
        est (object): The fitted FEAT estimator.
        id (int): The identifier for the specific model in the estimator's archive.
    Methods:
        __init__(est, id):
            Initializes the FeatPopEstimator with a fitted FEAT estimator
            and a model ID.
        fit(X, y):
            Dummy fit method to set the estimator as fitted.
        predict(X):
            Prepares the input data and predicts the output using the
            model from the estimator's archive.
        score(X, y):
            Computes the R^2 score of the prediction.
        model():
            Retrieves the model equation from the estimator's archive.
    """
    def __init__(self, est, id):
        self.est = est
        self.id  = id

    def fit(self, X, y):
        self.is_fitted_ = True

    def predict(self, X):
        
        X = self.est._prep_X(X)

        return self.est.cfeat_.predict_archive(self.id, X)

    def score(self, X, y):
        yhat = self.predict(X).flatten()
        return r2_score(y,yhat)
    
    def model(self):
        archive = self.est.cfeat_.get_archive(False)
        ind = [i for i in archive if i['id']==self.id][0]

        eqn = f"{np.round(ind['ml']['bias'], 5)}"
        for eq, w in zip(ind['eqn'].replace('[', '').split(']'), ind['w']):
            if str(w)[0]=='-':
                eqn = eqn + f'{np.round(float(w), 2)}*{eq}'
            else:
                eqn = eqn + f'+{np.round(float(w), 2)}*{eq}'

        return eqn

def model(est, X=None) -> str:
    """
    Return a sympy-compatible string of the final model. 

    Parameters
    ----------
    est: sklearn regressor
        The fitted model. 
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model. 

    Notes
    -----

    Ensure that the variable names appearing in the model are identical to 
    those in the training data, `X`, which is a `pd.Dataframe`. 
    If your method names variables some other way, e.g. `[x_0 ... x_m]`, 
    you can specify a mapping in the `model` function such as:

        ```
        def model(est, X):
            mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
            new_model = est.model_
            for k,v in mapping.items():
                new_model = new_model.replace(k,v)
        ```

    If you have special operators such as protected division or protected log,
    you will need to handle these to assure they conform to sympy format. 
    One option is to replace them with the unprotected versions. Post an issue
    if you have further questions: 
    https://github.com/cavalab/srbench/issues/new/choose
    """

    model_str = None
    if isinstance(est, FeatPopEstimator):
        model_str = est.model()
    else:
        model_str = est.cfeat_.get_eqn()

    # Here we replace "|" with "" to handle
    # protecte sqrt (expressed as sqrt(|.|)) in FEAT) 
    model_str = est.cfeat_.get_eqn()
    model_str = model_str.replace('|','')

    # use python syntax for exponents
    model_str = model_str.replace('^','**')

    return model_str


def get_population(est) -> list[RegressorMixin]:
    """
    Return the final population of the model. This final population should
    be a list with at most 100 individuals. Each of the individuals must
    be compatible with scikit-learn, so they should have a predict method.

    Also, it is expected that the `model()` function can operate with them,
    so they should have a way of getting a simpy string representation.
    
    Returns
    -------
    A list of scikit-learn compatible estimators that can be used for prediction.
    """

    # passing True will return just the front, and False will return final population
    archive = est.cfeat_.get_archive(False)

    pop = []

    # archive contains individuals serialized in json objects. let's get their ids
    for ind in archive:
        # Archive is sorted by complexity
        pop.append(
            FeatPopEstimator(est, ind['id'])
        )

        # Stopping here to avoid too many models
        if len(pop) >= 100:
            break


    return pop

def get_best_solution(est) -> RegressorMixin:
    """
    Return the best solution from the final model. 
    
    Returns
    -------
    A scikit-learn compatible estimator
    """

    return est


################################################################################
# Optional Settings
################################################################################


"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function. 
    Allows one to configure aspects of the training process.

Options 
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing. 
        for running the tests. called as 
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it. 
        if 0, use all training samples for fit. 
    scale_x: bool, default = True 
        Normalize the input data prior to fit. 
    scale_y: bool, default = True 
        Normalize the input label prior to fit. 
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit. 
        The function signature should be (est, X, y). 
            est: sklearn regressor; the fitted model. 
            X: pd.DataFrame; the training data. 
            y: training labels.
"""

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    if est.batch_size < len(X):
        est.gens = int(est.gens*len(X)/est.batch_size)
    print('FEAT gens adjusted to',est.gens)
    # adjust max dim
    est.max_dim=min(max(est.max_dim, X.shape[1]), 20)
    print('FEAT max_dim set to',est.max_dim)

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {'gens': 5,
                                  'pop_size': 10
                                 }
                  )
