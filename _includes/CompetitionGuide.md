[PR]: https://github.com/cavalab/srbench/compare/Competition2022...change-to-my-fork
[example]: https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/
[metadata]: https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/metadata.yml
[regressor]: https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/regressor.py
[env]: https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/environment.yml
[paper]: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c0c7c76d30bd3dcaefc96f40275bdc0a-Abstract-round1.html
[branch]: https://github.com/cavalab/srbench/tree/Competition2022

# SRBench 2022 Competition Guidelines
To participate, the steps are relatively straightforward. 
Participants fork this repo, add a method in the submission folder (see [submission/feat-example](https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/)), and submit it as a [pull request][PR] to the [Competition2022 branch][branch]. 
Once submitted, the continuous integration process will give feedback if there are any problems with the submission. 
Participants can then update their PR as necessary to have it pass the tests. 

Once everything is working, participants can basically sit tight: the competition organizers will then set about testing the methods on a variey of datasets. 

## Instructions

1. Fork this repository and clone it. Check out the [Competition2022 branch][branch].

    ```bash
    git clone git@github.com:your-name/srbench
    cd srbench
    git checkout Competition2022 # may need to "git fetch" first if this fails
    ```

2. Make a folder in the `submission/` named after your method. You can start with a template by copying the [`submission/feat-example`][example] folder, renaming and editing it. 

3. In the folder, put the contents of your submission. This folder should contain:

    1. `metadata.yml` (**required**): A file describing your submission, following the descriptions in [submission/feat-example/metadata.yml][metadata]. 
    2. `regressor.py` (**required**): a Python file that defines your method, named appropriately. See [submission/feat-example/regressor.py][regressor] for complete documentation. 
        It should contain:
        -   `est`: a sklearn-compatible `Regressor` object. 
        -   `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](###-returning-a-sympy-compatible-model-string). 
        -   `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to `evaluate_model.py`.
        
    3. `LICENSE` *(optional)* A license file
    4. `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
    See [submission/feat-example/environment.yml][env] for an example.
    It will be used to update the baseline environment (`environment.yml` in the root directory). 
    To the extent possible, conda should be used to specify the dependencies you need. 
    If your method is part of conda, great! You can just put that in here and leave `install.sh` blank. 
    5. `install.sh` *(optional)*: a bash script that installs your method. 
    **Note: scripts should not require sudo permissions. The library and include paths should be directed to conda environment; the environmental variable `$CONDA_PREFIX` specifies the path to the environment.
    6. additional files *(optional)*: you may include a folder containing the code for your method in the submission. Otherwise, `install.sh` should pull the source code remotely. 

    7. Commit your changes and submit your branch as a [pull request Competition 2022 branch][PR]. 

    8. Once the tests pass, you will be an official competitor!

### Version control

The install process should guarantee that the version for your algorithm that gets installed is **fixed**. 
You can do this in many ways: 

1. Including the source code with the submission. 
2. Pulling a tagged version/release of your code from a git repository. 
3. Checking out a specific commit, as in the provided example. 

## Regressor Guide

For detailed documentation, carefully read the example `regressor.py` file in [submission/feat-example/regressor.py][regressor], which describes several options for configuring your algorithm. 

### Returning a sympy compatible model string

In order to check for exact solutions to problems with known, ground-truth models, each SR method returns a model string that can be manipulated in [sympy](https://www.sympy.org). 
Assure the returned model meets these requirements:

1. The variable names appearing in the model are identical to those in the training data, `X`, which is a `pd.Dataframe`. 
If your method names variables some other way, e.g. `[x_0 ... x_m]`, you can
specify a mapping in the `model` function such as:

```python
def model(est, X):
    mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
    new_model = est.model_
    for k,v in mapping.items():
        new_model = new_model.replace(k,v)
```

2. The operators/functions in the model are available in [sympy's function set](https://docs.sympy.org/latest/modules/functions/index.html). 


### CLI methods

Is your SR method typically called via a command line interface? 
Check out this [gist](https://gist.github.com/folivetti/609bc9b854c51968ef90aa675ccaa60d) to make a Sklearn interface. 

## Competition Details

Once a method is successfully added to the Competition2022 branch, the organizing team will set about running each participant's method through a set of datasets. 
These datasets include synthetic and real-world tasks. 
Each dataset will have less than 10,000 samples and fewer than 100 features. 

### Stages

The competition consists of three stages, summarized in the table below. 

| Stage     | Filter  | Synthetic | Real-World |
|---|---|---|---|
| Benchmark Data | PMLB-20 | Synthetic | Real-World |
| Criteria  | Better than ElasticNet | Accuracy, Simplicity, Exact Solutions | Accuracy, Simplicity, Expert Assessment |


### Judging Criteria

Stay tuned for a detailed guide. 

### Accuracy

The [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html?highlight=r2_score#sklearn.metrics.r2_score)

#### Complexity

Complexity is defined as the number of nodes in the final model after it has been converted to a sympy object and symbolically simplified. 

### Exact Solutions

We check for _symbolic equivalence_ using sympy. 
See the definition in our [paper][paper] for details. 

### Expert Assessment

The real-world task will be one in which an expert model is available for comparison. 
In addition to the metrics above, participant solutions will be evaluated by a domain expert to determine their interpretability in the application context. 

### Computing Environment

Experiments will be run in a heterogeneous cluster computing environment composed of several hosts with 24-28 core Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz processors and 250 GB of RAM. 

The cluster uses an LSF scheduler. 
We define a _job_ as the training of an algorithm on a single dataset. 
For each job, an algorithm will have access to the following resources:

| Benchmark | PMLB-20 | Synthetic | Real-World |
|-----------|--------:|----------:|-----------:|
| RAM       | 16 GB   | 16 GB     | 16 GB      |
| CPU Cores | 4       | 8         | 8          |
| Wall-clock Time (H) | 24 | 1-10 | 10 |


CPU Cores for each job will span a single host, so methods are encouraged to support CPU-level parallelism. 

PMLB-20 is a set of 20 datasets we will select from the [Penn Machine Learning Benchmark](https://github.com/EpistasisLab/pmlb) used in our [original SRBench analysis](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers).  


### Time Budget

Methods must adhere to a fixed time budget for the competition. 
*These time budgets are designed to be **generous**, not **restrictive***. 
Our goal is to allow "slow" implementations enough time to learn. 

All datasets will be less than 10,000 rows and fewer than 100 features. 
The time limits are as follows:

- For datasets up to 1000 rows, 60 minutes (1 hour)
- For datasets up to 10000 rows, 600 minutes (10 hours)


If a call to `est.fit()` takes longer than the alotted time, it will receive
a SIGALRM signal and be terminated. Users may choose to handle this signal if
they wish in order to return an "any time" solution. 

**The preferred approach is that participants design their algorithms to converge in less than an hour for 1000x100 datasets, and less than 10 hours for 10000x100 datasets.**
Participants are also encouraged to include a `max_time` parameter and set it appropriately. 

To define dataset-specific runtime parameters, users can define a `pre_train()`
function in `regressor.py` as part of `eval_kwargs`. 
As an example, one could define the following in `regressor.py`:

```python
def pre_train_fn(est, X, y): 
    """set max_time in seconds based on length of X."""
    if len(X)<=1000:
        max_time = 360 
    else:
        max_time = 3600
    est.set_params(max_time )

# pass the function eval_kwargs
eval_kwargs = {
    'pre_train': pre_train_fn
}
```


### Hyperparameter Tuning

**Warning** If choose to conduct hyperparameter tuning, this counts towards the time limit. 
Make sure to set the `max_time` accordingly.  
{: .notice--warning}

Unlike our [prior study][paper], the competition does not automatically conduct hyper-parameter tuning for each method. 
Thus, it is up to participants to decide whether they want to include hyperparameter tuning as part of algorithm training in `regressor.py`. 
Here is an example of how to do so:

```python
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

hyper_params = { 'alpha': (1e-04,0.001,0.01,0.1,1,) }

# define your base estimator
base_est=linear_model.LassoLars()

# set est to be a GridSearchCV estimator
est = GridSearchCV(estimator=base_est, 
                   param_grid=hyper_params)

# addition definitions
# ...
```

In this example, the estimator is a GridSearchCV object wrapping a LassoLars model. 
During fit, it will tune the `alpha` parameter using cross-validation. 
For other hyper-parameter optimizers, see the [scikit-learn docs](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers).
