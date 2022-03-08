*Note: you are on the Competition2022 branch. 
View the [main SRBench page](https://github.com/cavalab/srbench)*

# SRBench 2022 Competition

# Guidelines

1. Fork this repository

2. Make a folder in the `submission/` named after your method. You can start with a template by copying the `submission/example` folder, renaming and editing it. 

3. In the folder, put the contents of your submission. This folder should contain:

    1. `metadata.yaml` (**required**): A file describing your submission, following the descriptions in `example/metadata.yaml`.  
    2. `regressor.py` (**required**): a Python file that defines your method, named appropriately. It should contain:
        -   `est`: a sklearn-compatible `Regressor` object. 
        -   `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](###-returning-a-sympy-compatible-model-string). 
        -   `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to [evaluate_model()](https://github.com/cavalab/srbench/blob/e3ba2c71dd08b1aaa76414a0af10411b98db59ee/experiment/evaluate_model.py#L24).
  See the [experiment/methods/AFPRegressor.py](https://github.com/cavalab/srbench/blob/master/experiment/methods/AFPRegressor.py) and/or other methods in that folder for examples.
    3. `LICENSE` *(optional)* A license file
    4. `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
    It will be used to update the baseline environment. 
    To the extent possible, conda should be used to specify the dependencies you need. 
    If your method is part of conda, great! You can just put that in here and leave `install.sh` blank. 
    5. `install.sh` *(optional)*: a bash script that installs your method. 
    6. additional files: you may include a folder containing the code for your method in the submission. Otherwise, `install.sh` should pull the source code remotely. 

### Version control

The install process should guarantee that the version for your algorithm that gets install is **fixed**. You can do this in many ways: 
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

