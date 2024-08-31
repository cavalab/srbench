Contribution Guide
==================

We are happy to accept contributions of methods, as well as updates to the benchmarking framework. 
Below we specify minimal requirements for contributing a method to this benchmark.

Ground Rules
=============

1. In general you should submit [pull requests](https://github.com/cavalab/srbench/compare) to the [dev branch](https://github.com/cavalab/srbench/tree/dev). 
2. Make the PR detailed and reference [specific issues](https://github.com/cavalab/srbench/issues) if the PR is meant to address any. 
3. **Please be kind and please be patient**. We will be, too.  

How to contribute an SR method
==============================

To contribute a symbolic regression method for benchmarking, fork the repo, make the changes listed below, and submit a pull request to the `dev` branch. 
Once your method passes the basic tests and we've reviewed it, congrats! 
We will plan to benchmark your method on hundreds of regression problems. 

Please note that the schedule for updating benchmarks is dependent on a lot of factors including availability of computing resources and availability of all our contributors. 
If you are on a tight schedule, it is better to plan to benchmark your method yourself. 
You can leverage this code base and previous experimental results to do so.

## Requirements

- An open-source method with a [scikit-learn compatible API](https://scikit-learn.org/stable/developers/develop.html)
- Your method should be compatible with **Python 3.7 or higher** to ensure compatibility with conda-forge.
- If your method uses a random seed, it should have a `random_state` attribute that can be set.
- Methods must have their own folders in the `algorithms` directory (e.g., `algorithms/feat`). 
This folder should contain:
  1. `metadata.yml` (**required**): A file describing your submission, following the descriptions in [algorithms/feat/metadata.yml][metadata]. 
  2. `regressor.py` (**required**): a Python file that defines your method, named appropriately. See [algorithms/feat/regressor.py][regressor] for complete documentation. 
      It should contain:
      -   `est`: a sklearn-compatible `Regressor` object. 
      -   `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](###-returning-a-sympy-compatible-model-string). 
      -   `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to `evaluate_model.py`.
      -   `get_population(est) --> List[RegressorMixin]`: a function that return a list of at most 100 expressions, if using pareto front, population-based optimization, beam search, or any strategy that allows your algorithm to explore several expressions. If this is not valid for your algorithm, you can just wrap the estimator in a list (_i.e._, `return [est]`). Every element from the returned list must be a compatible `Regressor`, meaning that calling `predict(X)` should work, as well as your custom `model(est, X=None)` method for getting a string representation.
      -   `get_best_solution(est)`: should provide an easy way of accessing the best solution from the current population, if this feature is valid for your algorithm. If not, then return the estimator itself `return est`.
  3. `LICENSE` *(optional)* A license file
  4. `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
  It will be used to update the baseline environment (`environment.yml` in the root directory). 
  To the extent possible, conda should be used to specify the dependencies you need. 
  If your method is part of conda, great! You can just put that in here and leave `install.sh` blank. 
  5. `requirements.txt` *(optional)*: a pypi requirements file. The script will run `pip install -r requirements.txt` if this file is found, before proceeding.
  5. `install.sh` *(optional)*: a bash script that installs your method. 
  **Note: scripts should not require sudo permissions. The library and include paths should be directed to conda environment; the environmental variable `$CONDA_PREFIX` specifies the path to the environment.
  6. **do not include your source code**. use `install.sh` to pull it from a stable source repository. 

### model compatibility with sympy

In order to check for exact solutions to problems with known, ground-truth models, each SR method returns a model string that can be manipulated in [sympy](https://www.sympy.org). 
Assure the returned model meets these requirements:

1. The variable names appearing in the model are identical to those in the training data, `X`, which is a `pd.Dataframe`. 
If your method names variables some other way, e.g. `[x_0 ... x_m]`, you can
specify a mapping in the `model` function such as:

```python
def model(est, X):
    mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
    new_model = est.model_
    for k,v in reversed(mapping.items()):
        new_model = new_model.replace(k,v)
```

2. The operators/functions in the model are available in [sympy's function set](https://docs.sympy.org/latest/modules/functions/index.html). 
