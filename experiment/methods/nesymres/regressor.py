import time
import json
import os, sys

import torch
import numpy as np
from sklearn import feature_selection 

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, BFGSParams
from functools import partial
import sympy as sp
from sympy import lambdify
import omegaconf

from tempfile import TemporaryDirectory

from sklearn.base import BaseEstimator, RegressorMixin

eq_setting = {
    "config": {
        "max_len": 20,
        "positive": True,
        "env_name": "eqlearn",
        "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:2",
        "max_ops": 5,
        "int_base": 10,
        "precision": 10,
        "rewrite_functions": "",
        "variables": [
            "x_1",
            "x_2",
            "x_3"
        ],
        "eos_index": 1,
        "pad_index": 0
    },
    "total_coefficients": [
        "cm_0",
        "cm_1",
        "cm_2",
        "cm_3",
        "cm_4",
        "cm_5",
        "cm_6",
        "cm_7",
        "cm_8",
        "cm_9",
        "cm_10",
        "cm_11",
        "cm_12",
        "cm_13",
        "cm_14",
        "cm_15",
        "cm_16",
        "cm_17",
        "cm_18",
        "cm_19",
        "cm_20",
        "cm_21",
        "cm_22",
        "cm_23",
        "cm_24",
        "cm_25",
        "cm_26",
        "cm_27",
        "cm_28",
        "cm_29",
        "cm_30",
        "cm_31",
        "cm_32",
        "cm_33",
        "cm_34",
        "cm_35",
        "cm_36",
        "cm_37",
        "cm_38",
        "cm_39",
        "ca_0",
        "ca_1",
        "ca_2",
        "ca_3",
        "ca_4",
        "ca_5",
        "ca_6",
        "ca_7",
        "ca_8",
        "ca_9",
        "ca_10",
        "ca_11",
        "ca_12",
        "ca_13",
        "ca_14",
        "ca_15",
        "ca_16",
        "ca_17",
        "ca_18",
        "ca_19",
        "ca_20",
        "ca_21",
        "ca_22",
        "ca_23",
        "ca_24",
        "ca_25",
        "ca_26",
        "ca_27",
        "ca_28",
        "ca_29",
        "ca_30",
        "ca_31",
        "ca_32",
        "ca_33",
        "ca_34",
        "ca_35",
        "ca_36",
        "ca_37",
        "ca_38",
        "ca_39"
    ],
    "total_variables": [
      "x_1",
      "x_2",
      "x_3"
    ],
    "word2id": {
      "0": 28,
      "1": 29,
      "2": 30,
      "3": 31,
      "4": 32,
      "5": 33,
      "x_1": 4,
      "x_2": 5,
      "x_3": 6,
      "abs": 7,
      "acos": 8,
      "add": 9,
      "asin": 10,
      "atan": 11,
      "cos": 12,
      "cosh": 13,
      "coth": 14,
      "div": 15,
      "exp": 16,
      "ln": 17,
      "mul": 18,
      "pow": 19,
      "sin": 20,
      "sinh": 21,
      "sqrt": 22,
      "tan": 23,
      "tanh": 24,
      "-3": 25,
      "-2": 26,
      "-1": 27,
      "P": 0,
      "S": 1,
      "F": 2,
      "c": 3
    },
    "id2word": {
      "1": "S",
      "2": "F",
      "3": "c",
      "4": "x_1",
      "5": "x_2",
      "6": "x_3",
      "7": "abs",
      "8": "acos",
      "9": "add",
      "10": "asin",
      "11": "atan",
      "12": "cos",
      "13": "cosh",
      "14": "coth",
      "15": "div",
      "16": "exp",
      "17": "ln",
      "18": "mul",
      "19": "pow",
      "20": "sin",
      "21": "sinh",
      "22": "sqrt",
      "23": "tan",
      "24": "tanh",
      "25": "-3",
      "26": "-2",
      "27": "-1",
      "28": "0",
      "29": "1",
      "30": "2",
      "31": "3",
      "32": "4",
      "33": "5"
    },
    "una_ops": [
      "asin",
      "cos",
      "exp",
      "ln",
      "pow2",
      "pow3",
      "pow4",
      "pow5",
      "sin",
      "sqrt",
      "tan"
    ],
    "bin_ops": [
      "asin",
      "cos",
      "exp",
      "ln",
      "pow2",
      "pow3",
      "pow4",
      "pow5",
      "sin",
      "sqrt",
      "tan"
    ],
    "rewrite_functions": [],
    "total_number_of_eqs": 200,
    "eqs_per_hdf": 200,
    "generator_details": {
      "max_len": 20,
      "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:2",
      "max_ops": 5,
      "rewrite_functions": "",
      "variables": [
        "x_1",
        "x_2",
        "x_3"
      ],
      "eos_index": 1,
      "pad_index": 0
    },
    "unique_index": None
}

cfg = None
with TemporaryDirectory() as temp_dir:
    cname = temp_dir + "/config.yaml"
    with open(cname, "w") as f:
        f.write("""
train_path: data/dataset/100000000
val_path: data/validation

raw_test_path: data/raw_datasets/150
test_path: data/validation
model_path: .../weights/nesymres_100M.ckpt

wandb: True
num_of_workers: 28
batch_size: 25
epochs: 20
val_check_interval: 0.02
precision: 16

dataset_train:
  total_variables: #Do not fill
  total_coefficients: #Do not fill
  max_number_of_points: 800  #2000 before
  type_of_sampling_points: logarithm
  predict_c: True
  fun_support:
    max: 10
    min: -10
  constants:
    num_constants: 3
    additive:
      max: 2
      min: -2
    multiplicative:
      max: 5
      min: 0.1

dataset_val:
  total_variables: #Do not fill
  total_coefficients: #Do not fill
  max_number_of_points: 500
  type_of_sampling_points: constant
  predict_c: True
  fun_support:
    max: 10
    min: -10
  constants:
    num_constants: 3
    additive:
      max: 2
      min: -2
    multiplicative:
      max: 5
      min: 0.1

dataset_test:
  total_variables: #Do not fill
  total_coefficients: #Do not fill
  max_number_of_points: 500
  type_of_sampling_points: constant
  predict_c: False
  fun_support:
    max: 10
    min: -10
  constants:
    num_constants: 3
    additive:
      max: 2
      min: -2
    multiplicative:
      max: 5
      min: 0.1

architecture:
  sinuisodal_embeddings: False
  dec_pf_dim: 512
  dec_layers: 5
  dim_hidden: 512 #512
  lr: 0.0001
  dropout: 0
  num_features: 10
  ln: True
  N_p: 0
  num_inds: 50
  activation: "relu"
  bit16: True
  norm: True
  linear: False
  input_normalization: False
  src_pad_idx: 0
  trg_pad_idx: 0
  length_eq: 60
  n_l_enc: 5
  mean: 0.5  
  std: 0.5 
  dim_input: 4
  num_heads: 8
  output_dim: 60

inference:
  beam_size: 2
  bfgs:
    activated: True
    n_restarts: 10
    add_coefficients_if_not_existing: False
    normalization_o: False
    idx_remove: True
    normalization_type: MSE
    stop_time: 3600

# @package _group_
hydra:
  run:
    dir: run/${dataset_train.predict_c}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
      dir: runs/${dataset_train.predict_c}/${now:%Y-%m-%d}/${now:%H-%M-%S}
""")
        
    cfg = omegaconf.OmegaConf.load(cname)


def get_top_k_features(X, y, k=10):
    if y.ndim==2:
        y=y[:,0]
    if X.shape[1]<=k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])

def get_variables(equation):
    """ Parse all free variables in equations and return them in
    lexicographic order"""

    expr = sp.parse_expr(equation)
    variables = expr.free_symbols
    variables = {str(v) for v in variables}

    # # Tighter sanity check: we only accept variables in ascending order
    # # to avoid silent errors with lambdify later.
    # if (variables not in [{'x'}, {'x', 'y'}, {'x', 'y', 'z'}]
    #         and variables not in [{'x1'}, {'x1', 'x2'}, {'x1', 'x2', 'x3'}]):
    #     raise ValueError(f'Unexpected set of variables: {variables}. '
    #                      f'If you want to allow this, make sure that the '
    #                      f'order of variables of the lambdify output will be '
    #                      f'correct.')

    # Make a sorted list out of variables
    # Assumption: the correct order is lexicographic (x, y, z)
    variables = sorted(variables)

    return variables


def evaluate_func(func_str, vars_list, X):
    assert X.ndim == 2
    assert len(set(vars_list)) == len(vars_list), 'Duplicates in vars_list!'

    order_list = vars_list
    indeces = [int(x[2:])-1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        f = lambdify([], func_str)
        return f() * np.ones(X.shape[0])

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    
    X_padded[:, :X.shape[1]] = X[:,:X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    X_subsel = X_padded[:, indeces]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    f = lambdify(vars_list, func_str)
    return f(*X_subsel.T)


n_restarts          = [1, 10, 20]
total_number_of_eqs = [100, 250, 500]
max_ops             = [10, 20, 50]

hyper_params = []
for restarts, eqs in zip(n_restarts, total_number_of_eqs):
    for ops in max_ops:
        hyper_params.append({
          'n_restarts' : [restarts],
          'total_number_of_eqs' : [eqs],
          'max_ops' : [ops]
        })
        
class NeSymResRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_restarts=10, total_number_of_eqs=200, max_ops=5,
                 weights_path="/srbench_pretrained/nesymres_100M.ckpt"):
        self.n_restarts = n_restarts
        self.total_number_of_eqs = total_number_of_eqs
        self.max_ops = max_ops
        self.weights_path = weights_path

    def fit(self, X, y):
        # Setting parameters
        cfg.inference.bfgs.n_restarts = self.n_restarts
        eq_setting['total_number_of_eqs'] = self.total_number_of_eqs
        eq_setting['config']['max_ops'] = self.max_ops

        ## Set up BFGS load rom the hydra config yaml
        bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

        # pre trained model supports at most 3
        self.top_k_features = get_top_k_features(X, y, k=3)
        X = X[:, self.top_k_features]

        eq_setting["total_variables"] = [f"x_{i+1}" for i in range(X.shape[1])]
        eq_setting["num_variables"] = int(X.shape[1])
        eq_setting["variables"] = [f"x_{i+1}" for i in range(X.shape[1])]

        print(eq_setting["total_variables"])

        params_fit = FitParams(word2id=eq_setting["word2id"], 
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                una_ops=eq_setting["una_ops"], 
                                bin_ops=eq_setting["bin_ops"], 
                                total_variables=eq_setting["total_variables"],  
                                total_coefficients=eq_setting["total_coefficients"],
                                rewrite_functions=eq_setting["rewrite_functions"],
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

        weights_path = self.weights_path
        if not os.path.isfile(weights_path):
          raise FileNotFoundError(weights_path)

        ## Load architecture, set into eval mode, and pass the config parameters
        #cfg.architecture.num_features = eq_setting["num_variables"]
        
        model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
        model.eval()

        if torch.cuda.is_available(): 
            model.cuda()

        fitfunc = partial(model.fitfunc, cfg_params=params_fit)

        output_ref = fitfunc(X,y) 

        self.model_skeleton_ = output_ref
        self.model_eq_       = model.get_equation()

        print("NeSymReS output ref: ", output_ref)
        print("NeSymReS Equation Skeleton: ", self.model_skeleton_)
        print("NeSymReS model_eq: ", self.model_eq_)

        self.pred_variables_ = get_variables(self.model_eq_[0])

        return self
    

    def predict(self, X):
        X = X[:, self.top_k_features]

        return evaluate_func(self.model_eq_[0],
                             self.pred_variables_, X)


est = NeSymResRegressor()

def model(est, X=None):
    return str(est.model_eq_[0])

# define eval_kwargs.
eval_kwargs = dict(use_dataframe=False)
