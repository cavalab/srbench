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
To handle the lack of a unified framework, we've specified minimal requirements for contributing a method to this benchmark: a scikit-learn compatible API for each method.

How to contribute
=================

- your open-source method
- a scikit-learn compatible API
- installation instructions
- a minimal script in `experiment/methods/` that specifies defines these items:
    -   `est`: a sklearn-compatible Regressor object 
    -   `hyper_params` : a dictionary or list of dictionaries specifying the hyperparameter search space
    -   `complexity`: a function that returns the complexity of the final model produced. 

## Complexity

To compare across methods with different representations, we need a precise definition of complexity. 

*Complexity* is defined as the total number of elements in the model, which in Koza-style GP would be the number of _nodes_ in the solution program tree. 
For example, the complexity of `(x + sin(3*y))` would be `len([x, y, 3, *, sin, +]) = 6`. 
Contributors are responsible for defining this within their provided method. 

