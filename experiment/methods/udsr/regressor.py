"""Borrows gpzgd's example of how to wrap a CLI learner with a sklearn interface
"""

from dso import DeepSymbolicRegressor
from dso import DeepSymbolicOptimizer

import sys
import os
from tempfile import TemporaryDirectory
import json
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

import tensorflow as tf

print("Num GPUs Available: ", tf.test.is_gpu_available())
print("CUDA Available:     ", tf.test.is_built_with_cuda())

function_set = [
    # ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt"],
    ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "const"],
    # ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"],
    ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "const", "poly"],
]
degree = [2, 3]
run_gp_meld = [True, False]

hyper_params = []
for f in function_set:
    hyper_params.append({
        'function_set' : [f],
        'degree'       : [2],
        'run_gp_meld'  : [True]
    })
        

class uDSRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42,
                 function_set=["add","sub","mul","div","sin","cos","exp","log","sqrt","const","poly"],
                 degree=3, run_gp_meld=True):
        
        self.random_state = random_state
        self.function_set = function_set
        self.degree       = degree
        self.run_gp_meld  = run_gp_meld

    def fit(self, X, y):
        # 1. create a temporary directory to store the training data set
        with TemporaryDirectory() as temp_dir:
            # 3. create a temp file and store the data
            X, y = check_X_y(X, y, accept_sparse=False)
            if len(y.shape) == 1:
                Z = np.hstack((X, y[:,None]))
            else:
                Z = np.hstack((X, y))

            fname   = temp_dir + "/tmpdata.csv"
            np.savetxt(f"{fname}", Z, delimiter=",")
            
            # 4. create a temp config file
            config = {
                'task' : {
                    'task_type' : 'regression',
                    'dataset'   : fname,
                    
                    # To customize a function set, edit this! See functions.py for a list of
                    # supported functions. Note "const" will add placeholder constants that
                    # will be optimized within the training loop. This will considerably
                    # increase runtime.
                    "function_set": self.function_set,
                
                    # Metric to be used for the reward function. See regression.py for
                    # supported metrics.
                    "metric" : "inv_nrmse",
                    "metric_params" : [1.0],
                
                    # Optional alternate metric to be used at evaluation time.
                    "extra_metric_test" : None,
                    "extra_metric_test_params" : [],
                
                    # NRMSE threshold for early stopping. This is useful for noiseless
                    # benchmark problems when DSO discovers the True solution.
                    "threshold" : 1e-12,
                
                    # With protected=False, floating-point errors (e.g. log of negative
                    # number) will simply returns a minimal reward. With protected=True,
                    # "protected" functions will prevent floating-point errors, but may
                    # introduce discontinuities in the learned functions.      
                    "protected" : True,
                
                    # You can add artificial reward noise directly to the reward function.
                    # Note this does NOT add noise to the dataset.
                    "reward_noise" : 0.0,
                    "reward_noise_type" : "r",
                    "normalize_variance" : False,
                
                    # Set of thresholds (shared by all input variables) for building
                    # decision trees. Note that no StateChecker will be added to Library
                    # if decision_tree_threshold_set is an empty list or null.
                    "decision_tree_threshold_set" : [],
                
                    # Parameters for optimizing the "poly" token.
                    # Note: poly_optimizer is turned on if and only if "poly" is in function_set.
                    "poly_optimizer_params" : {
                        # The (maximal) degree of the polynomials used to fit the data
                        "degree": self.degree,
                        # Cutoff value for the coefficients of polynomials. Coefficients
                        # with magnitude less than this value will be regarded as 0.
                        "coef_tol": 1e-6,
                        # linear models from sklearn: linear_regression, lasso,
                        # and ridge are currently supported, or our own implementation
                        # of least squares regressor "dso_least_squares".
                        "regressor": "dso_least_squares",
                        "regressor_params": {
                            # Cutoff value for p-value of coefficients. Coefficients with 
                            # larger p-values are forced to zero.
                            "cutoff_p_value": 1.0,
                            # Maximum number of terms in the polynomial. If more coefficients are nonzero,
                            # coefficients with larger p-values will be forced to zero.
                            "n_max_terms": None,
                            # Cutoff value for the coefficients of polynomials. Coefficients
                            # with magnitude less than this value will be regarded as 0.
                            "coef_tol": 1e-6
                        }
                    }
                },

                # Hyperparameters related to genetic programming hybrid methods.
                "gp_meld" : {
                    "run_gp_meld" : self.run_gp_meld,
                    "population_size" : 25,
                    "generations" : 25,
                    "crossover_operator" : "cxOnePoint",
                    "p_crossover" : 0.5,
                    "mutation_operator" : "multi_mutate",
                    "p_mutate" : 0.5,   
                    "tournament_size" : 5,
                    "train_n" : 50,
                    "mutate_tree_max" : 3,
                    "verbose" : True,
                    # Speeds up processing when doing expensive evaluations.
                    "parallel_eval" : False
                },

                # Only the key training hyperparameters are listed here. See
                # config_common.json for the full list.
                "training" : {
                    "n_samples" : 10000,
                    "batch_size" : 500,
                    "epsilon" : 0.02,
                    # Recommended to set this to as many cores as you can use! Especially if
                    # using the "const" token.
                    "n_cores_batch" : -1
                },
            
                # Only the key RNN controller hyperparameters are listed here. See
                # config_common.json for the full list.
                "controller" : {
                    "learning_rate": 0.0025,
                    "entropy_weight" : 0.03,
                    "entropy_gamma" : 0.7,
                    
                    # EXPERIMENTAL: Priority queue training hyperparameters.
                    "pqt" : True,
                    "pqt_k" : 10,
                    "pqt_batch_size" : 1,
                    "pqt_weight" : 200.0,
                    "pqt_use_pg" : False
                },
            
                # Hyperparameters related to including in situ priors and constraints. Each
                # prior must explicitly be turned "on" or it will not be used. See
                # config_common.json for descriptions of each prior.
                "prior": {
                    "length" : {
                        "min_" : 4,
                        "max_" : 100,
                        "on" : True
                    },
                    "inverse" : {
                        "on" : True
                    },
                    "trig" : {
                        "on" : True
                    },
                    "const" : {
                        "on" : True
                    },
                    "no_inputs" : {
                        "on" : True
                    },
                    "uniform_arity" : {
                        "on" : True
                    },
                    "soft_length" : {
                        "loc" : 10,
                        "scale" : 5,
                        "on" : True
                    },
                    "domain_range" : {
                        "on" : True
                    }
                },
                'experiment' : {
                    'logdir' : None,
                }
            }
            
            cname   = temp_dir + "/config.json"
            with open(cname, "w") as config_file:
                json.dump(config, config_file, indent=4)

            model = DeepSymbolicOptimizer(cname)
            train_result = model.train()

        self.program_ = train_result["program"]

        return self

    def predict(self, X):
        check_is_fitted(self, "program_")
        X = check_array(X)

        return self.program_.execute(X)


# using their sklearn wrapper (does not support GP steps)
# est = DeepSymbolicRegressor()
est = uDSRRegressor()


def model(est, X=None): # Should work for both methods
    # clean_pred_model from assess_symbolic model has fixes to the string rep
    return str(est.program_.sympy_expr)

# define eval_kwargs.
eval_kwargs = dict(use_dataframe=False)
    

if __name__ == "__main__":
    import numpy as np  
    
    # Generate some data
    np.random.seed(0)
    X = np.random.random((10, 2))
    y = np.sin(X[:,0]) + X[:,1] ** 2

    # Fit the model
    est.fit(X, y) # Should solve in ~10 seconds

    # View the best expression
    print(est.program_.pretty())
    print(est.program_.sympy_expr)

    # Is this counting "poly" as one node?
    print(len(est.program_.traversal))

    # Make predictions
    est.predict(2 * X)
