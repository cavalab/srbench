"""Borrows _HEAVILY_ Fabricio Olivetti de França's example of how to wrap a CLI learner with a sklearn interface
Output of the CLI command is in the format of:
xbar;std.dev;model;size;train_mse

where xbar and std.dev are numpy arrays containing the column means
and standard deviations of the training data (respectively), model the
stringified version of the evolved model, size is the
internally-computed size (number of nodes) of the model, and train_mse
is the internally computed MSE of the model on the training data

@Author: Fabricio Olivetti de França
@Date: 2020-01-05

"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error, r2_score

import os
from tempfile import TemporaryDirectory
import subprocess
import numpy as np

class GPZGD(BaseEstimator, RegressorMixin):

    def __init__(self, pop_size=200, generations=250,
                 tournament_size=3, validation_prop=0.0,
                 crossover_rate=0.3, sub_mutation_rate=0.4, point_mutation_rate=0.3, mutation_sigma=0.1,
                 min_tree_init=2, max_tree_init=4, max_tree_nodes=50,
                 opset="ADD,SUB,MUL,SIN,ERC,VAR",
                 learning_rate=0.01, learning_epochs=3, timeout=0, random_state=-1):
        """ Builds a Symbolic Regression using the cli interface of your algorithm.
        Examples
        --------
        >>> from gpzgd import GPZGD
        >>> import numpy as np
        >>> X = np.arange(100).reshape(100, 1)
        >>> y = x**2
        >>> reg = GPZGD(100, 100, 0.3, 0.7)
        >>> reg.fit(X, y)
        """

        self.validation_prop = validation_prop
        
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.opset = opset
        self.crossover_rate = crossover_rate
        self.sub_mutation_rate = sub_mutation_rate
        self.point_mutation_rate = point_mutation_rate
        self.mutation_sigma = mutation_sigma
        self.min_tree_init = min_tree_init
        self.max_tree_init = max_tree_init
        self.max_tree_nodes = max_tree_nodes
        self.learning_rate = learning_rate
        self.learning_epochs =learning_epochs
        self.timeout = timeout
        self.random_state = random_state

    def fit(self, X_train, y_train):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        # 1. create a temporary directory to store the training data set
        with TemporaryDirectory() as temp_dir:
            # 2. validate the consistency of the data matrices and create a single 2D array with X and y
            X_train, y_train = check_X_y(X_train, y_train, accept_sparse=False)
            if len(y_train.shape) == 1:
                Z_train = np.hstack((X_train, y_train[:,None]))
            else:
                Z_train = np.hstack((X_train, y_train))

            # 3. create a temp config file
            cname   = temp_dir + "/config"
            with open(cname, "w") as config_file:
                config_file.write(f"timeout={self.timeout}\n")
                config_file.write(f"validation_prop={self.validation_prop}\n")
                config_file.write(f"pop_size={self.pop_size}\n")
                config_file.write(f"generations={self.generations}\n")
                config_file.write(f"min_tree_init={self.min_tree_init}\n")
                config_file.write(f"max_tree_init={self.max_tree_init}\n")
                config_file.write(f"crossover_rate={self.crossover_rate}\n")
                config_file.write(f"point_mutation_rate={self.point_mutation_rate}\n")
                config_file.write(f"sub_mutation_rate={self.sub_mutation_rate}\n")
                config_file.write(f"max_tree_nodes={self.max_tree_nodes}\n")
                config_file.write(f"tournament_size={self.tournament_size}\n")
                config_file.write(f"mutation_sigma={self.mutation_sigma}\n")
                config_file.write(f"learning_rate={self.learning_rate}\n")
                config_file.write(f"learning_epochs={self.learning_epochs}\n")
                config_file.write(f"opset={self.opset}\n")
                config_file.write(f"elitism_rate=1\n")
                config_file.write(f"max_tree_depth=-1\n")
                config_file.write(f"standardise=Y\n")
                config_file.write(f"coef_op=Y\n")
                
            # 4. create a temp file and store the data
            fname   = temp_dir + "/tmpdata"
            np.savetxt(f"{fname}", Z_train, delimiter=" ", header=f"{Z_train.shape[0]} {Z_train.shape[1]}", comments="")

            # 5. call your cli binary with the parameters
            cwd = os.path.dirname(os.path.realpath(__file__))
            if self.random_state >= 0:
                ans = subprocess.check_output([ "bin/regressor", f"{fname}", f"{cname}", "-p", f"rng_seed={self.random_state}" ], cwd=cwd, universal_newlines=True)
            else:
                ans = subprocess.check_output([ "bin/regressor", f"{fname}", f"{cname}" ], cwd=cwd, universal_newlines=True)

        xbar, s, mdl, l, e = ans.split(";")

        self.xbar      = eval(xbar)
        self.s         = eval(s)
        self.expr      = mdl
        self.len       = int(l)
        self.cli_mse   = float(e)
        self.train_mse = mean_squared_error(y_train, self.eval_expr(X_train)) ## useful to compare with internal score!
        self.score     = r2_score(y_train, self.eval_expr(X_train))
        
        self.is_fitted_ = True

        return self

    def expr_str(self):
        return self.expr
    
    def eval_expr(self, X):
        """ Evaluates the expression with data point x. We assume that the expression is compatible with numpy """
        X = (X - self.xbar) / self.s
        y = eval(self.expr)

        # we can change any NaN or Inf to 0 to avoid evaluation error (not sure I like this, but okay)
        y[~np.isfinite(y)] = 0

        return y

    def predict(self, X_test, ic=None):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        check_is_fitted(self)
        X_test = check_array(X_test, accept_sparse=False)

        ##### TODO: Reverse Transform the predictions
        return self.eval_expr(X_test)

est = GPZGD()

# want to tune your estimator? wrap it in a sklearn CV class. 

def model(est, X=None):
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

    model_str = est.expr_str()
    model_str = model_str.replace("np.","").replace("[:,", "").replace("]", "")

    # use python syntax for exponents
    model_str = model_str.replace('^','**')

    if X is not None:
        mapping = { 'X'+str(i):k for i, k in enumerate(X.columns) }
        for k, v in reversed(mapping.items()):
            model_str = model_str.replace(k,v)

    return model_str

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
    if len(X) <= 1000:
        pop_size = 500
        generations = 5000
        max_time = 3564 ## 99% of one hour
    else:
        pop_size = 200
        generations = 5000
        max_time = 35640 ## 99% of ten hours

    est.tournament_size = 10
    est.pop_size = pop_size
    est.generations = generations
    est.timeout = max_time
    
# define eval_kwargs.
eval_kwargs = {
    "pre_train" : pre_train_fn
}
