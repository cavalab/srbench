from dsr import DeepSymbolicRegressor
import numpy as np
import copy
from .params._dsrregressor import params

base_config = {
   "task": {
      "task_type" : "regression",
      "name" : "srbench",
      "function_set": None,
      "dataset" : {
         "name" : None,
         "noise": None,
         "dataset_size_multiplier": 1.0
      },
      "metric" : "inv_nrmse",
      "metric_params" : [1.0],
      "threshold" : 1e-12,
      "protected" : False,
      "reward_noise" : 0.0
   },
   "prior": {
      "length" : {"min_" : 4, "max_" : 30},
      "repeat" : {"tokens" : "const", "max_" : 3},
      "inverse" : {},
      "trig" : {},
      "const" : {}
   },
   "training": {
      "logdir": "./log",
        "n_epochs": None,
        "n_samples": 500000,
        "batch_size": 1000,
        "complexity": "length",
        "complexity_weight": 0.0,
        "const_optimizer": "scipy",
        "const_params": {},
        "alpha": 0.5,
        "epsilon": 0.05,
        "verbose": True,
        "baseline": "R_e",
        "b_jumpstart": False,
        "n_cores_batch": 1,
        "summary": False,
        "debug": 0,
        "output_file": None,
        "save_all_r": False,
        "early_stopping": True,
        "pareto_front": False,
        "hof": 100
   },
   "controller": {
      "cell": "lstm",
      "num_layers": 1,
      "num_units": 32,
      "initializer": "zeros",
      "embedding": False,
      "embedding_size": 8,
      "optimizer": "adam",
      "learning_rate": 0.0005,
      "observe_action": False,
      "observe_parent": True,
      "observe_sibling": True,
      "entropy_weight": 0.005,
      "ppo": False,
      "ppo_clip_ratio": 0.2,
      "ppo_n_iters": 10,
      "ppo_n_mb": 4,
      "pqt": False,
      "pqt_k": 10,
      "pqt_batch_size": 1,
      "pqt_weight": 200.0,
      "pqt_use_pg": False,
      "max_length": 30
   },
   "gp": {
      "population_size": 1000,
      "generations": None,
      "n_samples" : 2000000,
      "tournament_size": 2,
      "metric": "nmse",
      "const_range": [
         -1.0,
         1.0
      ],
      "p_crossover": 0.95,
      "p_mutate": 0.03,
      "seed": 0,
      "early_stopping": True,
      "pareto_front": False,
      "threshold": 1e-12,
      "verbose": False,
      "protected": True,
      "constrain_const": True,
      "constrain_trig": True,
      "constrain_inv": True,
      "constrain_min_len": True,
      "constrain_max_len": True,
      "constrain_num_const": True,
      "min_length": 4,
      "max_length": 30,
      "max_const" : 3
   }
}

#double the evals
base_config['training']['n_samples'] = 1000000,

# Create the model
est = DeepSymbolicRegressor(base_config)

# est.set_params(**params)

# View the best expression
def model(est):
    return str(est.program_.sympy_expr)

def complexity(est):
    return len(est.program_.traversal)
