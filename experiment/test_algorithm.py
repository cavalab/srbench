import importlib
import json
import os
import sys
import types
from os.path import abspath
from os.path import dirname as d

import pytest

sys.path.append(d(abspath(__file__)))


from evaluate_model import evaluate_model
from read_file import read_file
from symbolic_utils import (
    complexity,
    round_floats,
)
from sympy import Symbol

# symbolic model stuff
from sympy.parsing.sympy_parser import parse_expr

if "OMP_NUM_THREADS" not in os.environ.keys():
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ.keys():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ.keys():
    os.environ["MKL_NUM_THREADS"] = "1"


# shared variables
dataset = "test/192_vineyard_small.tsv.gz"
results_path = "tmp_results"
random_state = 42


def get_algorithm(ml):

    algorithm = importlib.__import__(
        "methods." + ml + ".regressor", globals(), locals(), ["*"]
    )
    print("algorithm imported:", algorithm)
    return algorithm


def test_import(ml):
    """Check algorithm imports"""

    algorithm = get_algorithm(ml)
    assert "est" in dir(algorithm)
    assert "model" in dir(algorithm)

    eval_kwargs = {}
    if "eval_kwargs" in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs
        eval_kwarg_types = {
            "test_params": dict,
            "max_train_samples": int,
            "scale_x": bool,
            "scale_y": bool,
            "pre_train": types.FunctionType,
            "use_dataframe": bool,
        }
        for k, v in eval_kwargs.items():
            assert k in [
                "test_params",
                "max_train_samples",
                "scale_x",
                "scale_y",
                "pre_train",
                "use_dataframe",
            ]
            assert isinstance(v, eval_kwarg_types[k])


# TODO: need to update evaluate model to handle this format
def test_evaluate(ml):
    """Dataset evaluation"""
    algorithm = get_algorithm(ml)
    if "eval_kwargs" in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs
    else:
        eval_kwargs = {}
    print(eval_kwargs)
    json_file = evaluate_model(
        dataset=dataset,  # input file
        results_path=results_path,  # results directory
        random_state=random_state,  # random state seed
        est_name=ml,  # estimator name
        est=algorithm.est,  # estimator class
        algorithm=algorithm,
        model=algorithm.model,  # method to extract the model from the estimator
        test=True,  # testing
        **eval_kwargs,
    )
    print(json_file)
    print("hello")


@pytest.mark.order(after="test_evaluate")
def test_sympy(ml):
    """Sympy compatibility of model string"""

    algorithm = get_algorithm(ml)

    dataset_name = dataset.split("/")[-1][:-7]
    json_file = (
        results_path + "/" + dataset_name + "_" + ml + "_" + str(random_state) + ".json"
    )

    if os.path.exists(json_file):
        r = json.load(open(json_file, "r"))
    else:
        raise FileNotFoundError(json_file + " not found")

    r["algorithm"]

    raw_model = r["symbolic_model"]
    print("raw_model:", raw_model)
    X, labels, features = read_file(dataset)
    local_dict = {k: Symbol(k) for k in features}

    model_sym = parse_expr(raw_model, local_dict=local_dict)
    model_sym = round_floats(model_sym)
    print("sym model:", model_sym)

    # Should pass
    if hasattr(algorithm, "complexity"):
        try:
            model_complexity = algorithm.complexity(model_sym)
            print("model complexity (author provided):", model_complexity)
        except Exception as exc:
            print("algorithm complexity failed; using fallback:", exc)
            model_complexity = complexity(model_sym)
            print("model complexity (using our implementation):", model_complexity)
    else:  # Given that the string is sympy compatible, this should work
        model_complexity = complexity(model_sym)
        print("model complexity (using our implementation):", model_complexity)
