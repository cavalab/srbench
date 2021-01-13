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
Here are the requirements:

- an open-source method
- a scikit-learn compatible API
- a bash install script in `experiment/methods/src/` named `your-method_install.sh` that installs your method. 
  See [ellyn_install.sh](experiment/methods/src/ellyn_install.sh) as an example. 
  Our [Github actions workflow](.github/workflows/test.yml) will automatically recognize it. 
- a minimal script in `experiment/methods/` that defines these items:
    -   `est`: a sklearn-compatible `Regressor` object 
    -   `hyper_params` : a dictionary or list of dictionaries specifying the hyperparameter search space
    -   `complexity(est)`: a function that returns the complexity of the final model produced
    -   `model(est)`: a function that returns the form of the final model as a string 
  See [experiment/methods/afp.py](experiment/methods/afp.py) for an example.

**Complexity**: To compare across methods with different representations, we use this common definition of complexity. 
Contributors are responsible for defining this complexity within their provided method. 
Complexity is defined as the total number of elements in the model, which in Koza-style GP would be the number of _nodes_ in the solution program tree. 
For example, the complexity of `(x + sin(3*y))` would be `len([+, x, sin, *, 3, y]) = 6`. 
In other words, **every instance of basic math operators `(+, -, *, /, sin, cos, exp, log, %)`, constants, and input features should count toward the complexity**. 
So, if your method uses very complex operations, for example, an operator that is defined as `Op(x,y) = (x+sin(3*y))`, you need to decompose such operators into their component parts when accounting for this complexity. 
Note that the relative complexity of these basic math operators is not captured by this measure.
If you are unsure about this definition, please open a discussion ticket.

