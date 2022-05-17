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

    def __init__(self, pop_size=200, generations=250, K=3, validation_prop=0.0,
                 pc=0.3, pm=0.4, pp=0.3, mutation_scale=0.1,
                 min_init_depth=2, max_init_depth=4, max_size=50,
                 opset="ADD,SUB,MUL,SIN,ERC,VAR",
                 learning_rate=0.01, learning_epochs=3, max_time=0, random_state=-1):
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
        self.tournament_size = K
        self.opset = opset
        self.crossover_rate = pc
        self.sub_mutation_rate = pm
        self.point_mutation_rate = pp
        self.mutation_sigma = mutation_scale
        self.min_tree_init = min_init_depth
        self.max_tree_init = max_init_depth
        self.max_tree_nodes = max_size
        self.learning_rate = learning_rate
        self.learning_epochs =learning_epochs
        self.timeout = max_time
        if random_state >= 0:
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
            if hasattr(self, "random_state"):
                ans = subprocess.check_output([ "dist/regressor", f"{fname}", f"{cname}", "-p", f"rng_seed={self.random_state}" ], cwd=cwd, universal_newlines=True)
            else:
                ans = subprocess.check_output([ "dist/regressor", f"{fname}", f"{cname}" ], cwd=cwd, universal_newlines=True)

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
