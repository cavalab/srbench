[PR]: (https://github.com/cavalab/srbench/compare/Competition2022...?template=competition_template.md)

# SRBench 2022 Competition Guidelines
To participate, the steps are relatively straightforward. 
Participants fork this repo, add a method in the submission folder (see [submission/feat-example](https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/)), and submit it as a [pull request][PR] to the [Competition 2022 branch](https://github.com/cavalab/srbench/tree/Competition2022). 
Once submitted, the continuous integration process will give feedback if there are any problems with the submission. 
Participants can then update their PR as necessary to have it pass the tests. 

Once everything is working, participants can basically sit tight: the competition organizers will then set about testing the methods on a variey of datasets. 

## Instructions

1. Fork this repository and clone it. 

2. Make a folder in the `submission/` named after your method. You can start with a template by copying the `submission/example` folder, renaming and editing it. 

3. In the folder, put the contents of your submission. This folder should contain:

    1. `metadata.yml` (**required**): A file describing your submission, following the descriptions in `example/metadata.yml`.  
    2. `regressor.py` (**required**): a Python file that defines your method, named appropriately. See [submission/feat-example/regressor.py](https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/regressor.py) for complete documentation. 
        It should contain:
        -   `est`: a sklearn-compatible `Regressor` object. 
        -   `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](###-returning-a-sympy-compatible-model-string). 
        -   `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to `evaluate_model.py`.
        
    3. `LICENSE` *(optional)* A license file
    4. `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
    It will be used to update the baseline environment (`environment.yml` in the
    root directory). 
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

### Regressor Settings

#### CLI methods

Is your SR method typically called via a command line interface? 
Check out this [gist](https://gist.github.com/folivetti/609bc9b854c51968ef90aa675ccaa60d) to make a Sklearn interface. 

#### Time Budget

Methods must adhere to a fixed time budget for the competition. 
All datasets will be less than 10,000 rows and fewer than 100 features. 
The time limits are as follows:

- For datasets up to 1000 rows, 60 minutes
- For datasets up to 10000 rows, 600 minutes (10 hours)


If a call to `est.fit()` takes longer than the alotted time, it will receive
a SIGALRM signal and be terminated. Users may choose to handle this signal if
they wish in order to return an "any time" solution. 

The preferred solution is that participants include a `max_time` parameter and 
set it appropriately. 
To define dataset-specific runtime parameters, users can define a `pre_train()`
function in `regressor.py` as part of `eval_kwargs`. 
As an example, one could define the following in `regressor.py`:

```python
def pre_train_fn(est, X, y): 
    # set max_time in seconds based on length of X
    est.set_params( max_time = 360 if len(X)<=1000 else 3600 )

eval_kwargs = {
    'pre_train': pre_train_fn
}
```

Note that if choose to conduct hyperparameter tuning, this counts towards
    the time limit. Make sure to set the `max_time` accordingly.  
