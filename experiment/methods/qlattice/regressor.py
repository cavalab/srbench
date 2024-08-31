import signal

import feyn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight
from sympy.printing.printer import Printer


class InternalTimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print(f"raising InternalTimeOutException")
    raise InternalTimeOutException


def auto_run_time(ql,
                  data,
                  output_name,
                  kind="regression",
                  stypes=None,
                  n_epochs=200,
                  threads="auto",
                  max_complexity=10,
                  query_string=None,
                  loss_function=None,
                  criterion="wide_parsimony",
                  sample_weights=None,
                  function_names=None,
                  starting_models=None,
                  max_time=None
                  ):
    if max_time:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(max_time)

    feyn.validate_data(data, kind, output_name, stypes)

    if n_epochs <= 0:
        raise ValueError("n_epochs must be 1 or higher.")

    if threads == "auto":
        threads = feyn.tools.infer_available_threads()
    elif isinstance(threads, str):
        raise ValueError("threads must be a number, or string 'auto'.")

    models = []
    if starting_models is not None:
        models = [m.copy() for m in starting_models]
    m_count = len(models)

    priors = feyn.tools.estimate_priors(data, output_name)
    ql.update_priors(priors)

    try:
        for epoch in range(1, n_epochs + 1):
            new_sample = ql.sample_models(
                data,
                output_name,
                kind,
                stypes,
                max_complexity,
                query_string,
                function_names,
            )
            models += new_sample
            m_count += len(new_sample)

            models = feyn.fit_models(
                models,
                data=data,
                loss_function=loss_function,
                criterion=criterion,
                n_samples=None,
                sample_weights=sample_weights,
                threads=threads,
            )
            models = feyn.prune_models(models)
            ql.update(models)

        best = feyn.get_diverse_models(models)

    except InternalTimeOutException:
        print('InternalTimeOutException raised')
        best = feyn.get_diverse_models(models)

    return best


def numpy_check_to_DataFrame(X, y=None):
    if isinstance(X, np.ndarray):
        columns = [f'x__{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)

    if y is None:
        return X
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = pd.Series(y, name='target')
        elif y.ndim == 2:
            y = pd.DataFrame(y, columns=['target'])
        return X, y


def feature_names(X, y=None):
    if hasattr(X, 'columns'):
        input_names = X.columns
    else:
        input_names = None

    if y is None:
        return input_names
    else:
        if hasattr(y, 'name'):
            output_name = y.name
        else:
            output_name = None
        return input_names, output_name


class QLatticeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 kind='regression',
                 stypes=None,
                 n_epochs=200,
                 threads='auto',
                 max_complexity=10,
                 query_string=None,
                 loss_function=None,
                 criterion='wide_parsimony',
                 sample_weights=None,
                 function_names=None,
                 starting_models=None,
                 random_state=None,
                 max_time=None
                 ):
        self.kind = kind
        self.stypes = stypes
        self.n_epochs = n_epochs
        self.threads = threads
        self.max_complexity = max_complexity
        self.query_string = query_string
        self.loss_function = loss_function
        self.criterion = criterion
        self.sample_weights = sample_weights
        self.function_names = function_names
        self.starting_models = starting_models
        self.random_state = random_state
        self.max_time = max_time

    def fit(self, X, y, sample_weight=None):

        input_names, output_name = feature_names(X, y)

        X, y = check_X_y(X, y, y_numeric=True)
        X, y = numpy_check_to_DataFrame(X, y)
        data = pd.concat([X, y], axis=1)

        if input_names is not None:
            data = data.rename(columns=dict(zip(X.columns, list(input_names))))
        if output_name is not None:
            data = data.rename(columns={y.name: output_name})

        data.columns = data.columns.astype(str)

        if self.random_state is None:
            rseed = -1
        else:
            rseed = self.random_state

        if sample_weight is None:
            sample_weight = self.sample_weights
        else:
            sample_weight = _check_sample_weight(sample_weight, X)

        ql = feyn.QLattice(random_seed=rseed)
        self.models_ = auto_run_time(ql=ql,
                                     data=data,
                                     output_name=data.columns[-1],
                                     kind=self.kind,
                                     stypes=self.stypes,
                                     n_epochs=self.n_epochs,
                                     threads=self.threads,
                                     max_complexity=self.max_complexity,
                                     query_string=self.query_string,
                                     loss_function=self.loss_function,
                                     criterion=self.criterion,
                                     sample_weights=sample_weight,
                                     function_names=self.function_names,
                                     starting_models=self.starting_models,
                                     max_time=self.max_time
                                     )
        return self

    def predict(self, X, n=0):

        check_is_fitted(self)

        input_names = feature_names(X)
        X = check_array(X)

        X = numpy_check_to_DataFrame(X)
        if input_names is not None:
            X = X.rename(columns=dict(zip(X.columns, list(input_names))))
        X.columns = X.columns.astype(str)
        return self.models_[n].predict(X)

    def score(self, X, y, n=0):
        input_names, output_name = feature_names(X, y)
        X, y = check_X_y(X, y, y_numeric=True)
        X, y = numpy_check_to_DataFrame(X, y)
        data = pd.concat([X, y], axis=1)

        if input_names is not None:
            data = data.rename(columns=dict(zip(X.columns, list(input_names))))
        if output_name is not None:
            data = data.rename(columns={y.name: output_name})

        data.columns = data.columns.astype(str)

        return self.models_[n].r2_score(data)


est = QLatticeRegressor(
    kind='regression',
    n_epochs=200,
    max_complexity=10,
    criterion='wide_parsimony',
)


# do we need to make sure features have the same name as in original data?
def model(est, X):
    printer = Printer()
    string_model = printer.doprint(est.models_[0].sympify())

    try:
        mapping = {feyn.tools._sympy.get_sanitized_name(col): col for col in X.columns}
        for k, v in reversed(mapping.items()):
            string_model = string_model.replace(k, v)
    except:
        pass

    return string_model


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


def pre_train_fn(est, X, y):
    """set max_time in seconds based on length of X."""
    if len(X) <= 1000:
        max_time = 3600 - 5
    else:
        max_time = 36000 - 5
    est.set_params(max_time=max_time)


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=pre_train_fn,
    test_params={'n_epochs': 2,
                 }
)
