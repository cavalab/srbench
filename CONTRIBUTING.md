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
Once your method passes the basic tests and we've reviewed it, it will be available as part of srbench.

Please note that the schedule for updating benchmarks is dependent on a lot of factors including availability of computing resources and availability of all our contributors. 
If you are on a tight schedule, it is better to plan to benchmark your method yourself. 
You can leverage this code base and previous experimental results to do so.

## Requirements

- An open-source method with a [scikit-learn compatible API](https://scikit-learn.org/stable/developers/develop.html)
- Your method should be compatible with **Python 3.7 or higher** to ensure compatibility with conda-forge.
- If your method uses a random seed, it should have a `random_state` attribute that can be set.

### Where your files go

A submission spans **two** directories, and both are required.
This is the single most common thing to get wrong, so it is worth reading closely.

```
algorithms/<your-method>/          # how to INSTALL your method
├── metadata.yml                   #   required
├── environment.yml                #   optional
├── requirements.txt               #   optional
├── install.sh                     #   optional
├── Dockerfile                     #   optional
└── LICENSE                        #   optional

experiment/methods/<your-method>/  # how to CALL your method
├── regressor.py                   #   required
└── __init__.py                    #   required (an empty file)
```

The split follows from how the benchmark runs.
`algorithms/<your-method>/` is copied into your Docker image when it is built, so it holds everything needed to *install* your method.
`experiment/` is mounted into the running container, so `regressor.py` is read at *run* time and is never baked into the image.

Use the **same directory name** in both places.
Note that `metadata.yml` belongs with the install files in `algorithms/`, not next to `regressor.py`.

You can check your layout before opening a PR:

```bash
python scripts/check_method_layout.py
```

CI runs this same check, and it will fail your PR if anything is out of place.

#### `algorithms/<your-method>/`

  1. `metadata.yml` (**required**): A file describing your submission, following the descriptions in [algorithms/feat/metadata.yml][metadata]. Please fill in `name`, `authors`, `email`, `description` and `url`.
  2. `LICENSE` *(optional)* A license file
  3. `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission.
  It will be used to update the baseline environment (`base_environment.yml` in the root directory).
  To the extent possible, conda should be used to specify the dependencies you need.
  If your method is part of conda, great! You can just put that in here and leave `install.sh` blank.
  4. `requirements.txt` *(optional)*: a pypi requirements file. The script will run `pip install -r requirements.txt` if this file is found, before proceeding.
  5. `install.sh` *(optional)*: a bash script that installs your method.
  **Note: scripts should not require sudo permissions. The library and include paths should be directed to conda environment; the environmental variable `$CONDA_PREFIX` specifies the path to the environment.
  6. `Dockerfile` *(optional)*: we will try to dockerize all algorithms. You can optionally have a `Dockerfile` inside your `algorithms/your-submission` folder to describe specific images for running your algorithm. If no file is provided, it will use `alg-Dockerfile` for your container. You can specify the image as you like, as long as you have as minimal dependences the python packages described in `base_environment.yml`, as they are used to run the experiment scripts. See [this example](algorithms/tir/Dockerfile) in case you want to use a custom image. *Notice that there is a workflow to build the docker images and push them to dockerhub*.
  7. **do not include your source code**. use `install.sh` to pull it from a stable source repository.

#### `experiment/methods/<your-method>/`

  1. `regressor.py` (**required**): a Python file that defines your method. See [experiment/methods/feat/regressor.py][regressor] for complete documentation.
      It should contain:
      -   `est`: a sklearn-compatible `Regressor` object.
      -   `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](#model-compatibility-with-sympy).
      -   `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to `evaluate_model.py`. Only these keys are recognized: `test_params`, `max_train_samples`, `scale_x`, `scale_y`, `pre_train`, `use_dataframe`.
      -   We expect your algorithm to have a `max_time` parameter that lets us control the maximum execution time in seconds. When running the experiments in a cluster, we will give extra time to compensate for the overhead of initializing everything, and the maximum time considered is just the fit process. A signal `signal.SIGALRM` will be sent to your process if `fit(X, y)` exceeds the maximum time, and you can implement strategies to handle this signal. One idea is to store a random initial solution as the best and update it during the execution to ensure the `evaluate_model.py` script will find an equation to work on.
      -   The harness looks for the time limit under any of these attribute names: `max_time`, `timeout_in_seconds`, `timeout`, `stop_time`, `time_limit`. If your estimator exposes none of them, it will simply be killed when it runs long.
  2. `__init__.py` (**required**): an empty file, so the harness can import your method.

### Testing your submission locally

Build your image and run the same tests CI runs:

```bash
bash scripts/make_docker_compose_file.sh   # regenerates docker-compose.yml
docker compose build base                  # the shared base image, needed once
docker compose build <your-method>
docker compose run --rm <your-method> bash test.sh
```

### model compatibility with sympy

In order to check for exact solutions to problems with known, ground-truth models, each SR method returns a model string that can be manipulated in [sympy](https://www.sympy.org). 
Assure the returned model meets these requirements:

1. The variable names appearing in the model are identical to those in the training data, `X`, which is a `pd.Dataframe`. 
If your method names variables some other way, e.g. `[x_0 ... x_m]`, you can
specify a mapping in the `model` function such as:

```python
def model(est, X=None):
    mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
    new_model = est.model_
    for k,v in reversed(mapping.items()):
        new_model = new_model.replace(k,v)
```

2. The operators/functions in the model are available in [sympy's function set](https://docs.sympy.org/latest/modules/functions/index.html). 

[metadata]: https://github.com/cavalab/srbench/blob/master/algorithms/feat/metadata.yml
[regressor]: https://github.com/cavalab/srbench/blob/master/experiment/methods/feat/regressor.py

