Contribution Guide
==================

We are happy to accept contributions of methods, as well as updates to the benchmarking framework. 
Below we specify minimal requirements for contributing a method to this benchmark.

Ground Rules
=============

1. In general you should submit [pull requests](https://github.com/cavalab/srbench/compare) to the [dev branch](https://github.com/cavalab/srbench/tree/dev). 
2. Make the PR detailed and reference [specific issues](https://github.com/cavalab/srbench/issues) if the PR is meant to address any. 
3. Please be kind and please be patient. We will be, too.  

How to contribute an SR method
==============================

To contribute a symbolic regression method for benchmarking, fork the repo, make the changes listed below, and submit a pull request to the `dev` branch. 
Once your method passes the basic tests and we've reviewed it, congrats! 
We will plan to benchmark your method on hundreds of regression problems. 

## Requirements

1. An open-source method with a [scikit-learn compatible API](https://scikit-learn.org/stable/developers/develop.html)
2. If your method uses a random seed, it should have a `random_state` attribute that can be set.
3. If your method is installable via pip or conda, add it to the [environment file](https://github.com/cavalab/srbench/blob/master/environment.yml). 
  Otherwise, a bash install script in `experiment/methods/src/` named `your-method_install.sh` that installs your method. 
  See [ellyn_install.sh](https://github.com/cavalab/srbench/blob/master/experiment/methods/src/ellyn_install.sh) as an example. 
  Our [Github actions workflow](https://github.com/cavalab/srbench/blob/master/.github/workflows/test.yml) will automatically recognize it. 
4. A minimal script in `experiment/methods/` that defines these items:
    -   `est`: a sklearn-compatible `Regressor` object 
    -   `hyper_params` : a dictionary or list of dictionaries specifying the hyperparameter search space
    -   `model(est)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model.
    -   (optional): a dictionary named `eval_kwargs` that can specify method-specific arguments to [evaluate_model()](https://github.com/cavalab/srbench/blob/e3ba2c71dd08b1aaa76414a0af10411b98db59ee/experiment/evaluate_model.py#L24).
  See the [experiment/methods/AFPRegressor.py](https://github.com/cavalab/srbench/blob/master/experiment/methods/AFPRegressor.py) and/or other methods in that folder for examples.

### Returning a sympy compatible model string
In order to check for exact solutions to problems with known, ground-truth models, each SR method returns a model string that can be manipulated in [sympy](https://www.sympy.org). 
Assure the returned model meets these requirements:

1. The variable names appearing in the model are identical to those in the training data.
2. The operators/functions in the model are available in [sympy's function set](https://docs.sympy.org/latest/modules/functions/index.html). 
If they are not, they need to be defined in the model's script and referenced appropriately in the `model()` function, so that sympy can find them. 

We are still working out how to handle operators uniformly and appropriately, and currently rely on [experiment/symbolic_utils.py](https://github.com/EpistasisLab/srbench/blob/master/experiment/symbolic_utils.py) to post-process models.  
However, new additions to the repo should not require post-processing for compatibility.
See issue #58 for more details.
