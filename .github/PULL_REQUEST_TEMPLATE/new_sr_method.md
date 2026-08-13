<!--
Thanks very much for submitting a new method to SRBench!

Please check the list below and make sure you have everything covered. 
If you need help with the PR, feel free to tag @srbench-comp and we'll respond asap. 

-->
## Submission Checklist

A submission spans **two** directories, using the same name in both. See
[CONTRIBUTING.md](../../CONTRIBUTING.md#where-your-files-go) for the details.

- [ ] title of this PR is meaningful, i.e. "adding method X"

**`algorithms/<your-method>/`** — how to install your method

- [ ] `metadata.yml` (**required**): A file describing your submission, following the descriptions in [algorithms/feat/metadata.yml][metadata]. `name`, `authors`, `email`, `description` and `url` are filled in.
- [ ] `LICENSE` *(optional)* A license file
- [ ] `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission.
  It will be used to update the baseline environment (`base_environment.yml` in the root directory).
  To the extent possible, conda should be used to specify the dependencies you need.
  If your method is part of conda, great! You can just put that in here and leave `install.sh` blank.
- [ ] `requirements.txt` *(optional)*: a pypi requirements file. The script will run `pip install -r requirements.txt` if this file is found, before proceeding.
- [ ] `install.sh` *(optional)*: a bash script that installs your method **without sudo permissions**.
- [ ] I did not include source code; instead I used `install.sh` to pull it from a stable source repository.

**`experiment/methods/<your-method>/`** — how to call your method

- [ ] `regressor.py` (**required**): a Python file that defines your method. See [experiment/methods/feat/regressor.py][regressor] for complete documentation.
  `regressor.py` contains:
  - [ ] `est`: a sklearn-compatible `Regressor` object.
  - [ ] `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument.
  - [ ] `eval_kwargs` *(optional)*: a dictionary that can specify method-specific arguments to `evaluate_model.py`.
- [ ] `__init__.py` (**required**): an empty file, so the harness can import your method.

**Checks**

- [ ] `python scripts/check_method_layout.py` passes.
- [ ] I locally tested my method with:
      ```
      bash scripts/make_docker_compose_file.sh
      docker compose build base
      docker compose build <your-method>
      docker compose run --rm <your-method> bash test.sh
      ```

[metadata]: https://github.com/cavalab/srbench/blob/master/algorithms/feat/metadata.yml
[regressor]: https://github.com/cavalab/srbench/blob/master/experiment/methods/feat/regressor.py 