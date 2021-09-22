Contribution Guide
==================

The methods for symbolic regression have come a long way since the days of Koza-style genetic programming (GP).
Our goal with this project is to keep a living benchmark of modern symbolic regression, in the context of state-of-the-art ML methods.
Currently these are the challenges, as we see it, to achieving this:

- Lack of cross-pollination between the GP community and the ML community (different conferences, journals, societies etc)
- Lack of strong benchmarks in GP literature (small problems, toy datasets, weak comparator methods)
- Lack of a unified framework for SR, or GP

We are addressing the lack of pollination by making these comparisons open source, reproduceable and public, and hoping to share them widely with the entire ML research community.
We are trying to address the lack of strong benchmarks by providing open source benchmarking of many SR methods on large sets of problems, with strong baselines for comparison. 
To handle the lack of a unified framework, we've specified minimal requirements for contributing a method to this benchmark: a scikit-learn compatible API.

How to contribute
=================

To contribute a symbolic regression method for benchmarking, fork the repo, make the changes listed below, and submit a pull request. 
Once your method passes the basic tests and we've reviewed it, congrats! 
We will plan to benchmark your method on hundreds of regression problems. 

## Requirements

1. An open-source method with a [scikit-learn compatible API](https://scikit-learn.org/stable/developers/develop.html)
2. If your method uses a random seed, it should have a `random_state` attribute that can be set.
3. If your method is installable via pip or conda, add it to the [environment file](environment.yml). 
  Otherwise, a bash install script in `experiment/methods/src/` named `your-method_install.sh` that installs your method. 
  See [ellyn_install.sh](experiment/methods/src/ellyn_install.sh) as an example. 
  Our [Github actions workflow](.github/workflows/test.yml) will automatically recognize it. 
4. A minimal script in `experiment/methods/` that defines these items:
    -   `est`: a sklearn-compatible `Regressor` object 
    -   `hyper_params` : a dictionary or list of dictionaries specifying the hyperparameter search space
    -   `complexity(est)`: a function that returns the complexity of the final model produced (see below)
    -   `model(est)`: a function that returns a [**sympy-compatible**](www.sympy.org) string specifying the final model.
  See [experiment/methods/AFPRegressor.py](experiment/methods/AFPRegressor.py) for an example.

### Defining Complexity
Contributors are responsible for defining this complexity within their provided method. 
To compare across methods with different representations, we use this common definition of complexity. 
Complexity is defined as the total number of elements in the model, which in Koza-style GP would be the number of _nodes_ in the solution program tree. 
For example, the complexity of `(x + sin(3*y))` would be `len([+, x, sin, *, 3, y]) = 6`. 
In other words, **every instance of basic math operators `(+, -, *, /, sin, cos, exp, log, %)`, constants, and input features should count toward the complexity**. 
So, if your method uses very complex operations, for example, an operator that is defined as `Op(x,y) = (x+sin(3*y))`, you need to decompose such operators into their component parts when accounting for this complexity. 
Note that the relative complexity of these basic math operators is not captured by this measure.
If you are unsure about this definition, please open a discussion ticket.

### Returning a sympy compatible model string
In order to check for exact solutions to problems with known, ground-truth models, each SR method returns a model string that can be manipulated in [sympy](www.sympy.org). 
Assure the returned model meets these requirements:

1. The variable names appearing in the model are identical to those in the training data.
2. The operators/functions in the model are available in [sympy's function set](https://docs.sympy.org/latest/modules/functions/index.html). 
If they are not, they need to be defined in the model's script and referenced appropriately in the `model()` function, so that sympy can find them. 

We are still working out how to handle operators uniformly and appropriately, and currently rely on [experiment/symbolic_utils.py](https://github.com/EpistasisLab/srbench/blob/master/experiment/symbolic_utils.py) to post-process models.  
However, new additions to the repo should not require post-processing for compatibility.
See issue #58 for more details.