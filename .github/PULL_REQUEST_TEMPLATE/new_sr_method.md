<!--
Thanks very much for submitting a new method to SRBench!

Please check the list below and make sure you have everything covered. 
If you need help with the PR, feel free to tag @srbench-comp and we'll respond asap. 

-->
## Submission Checklist

- [ ] title of this PR is meaningful, i.e. "adding method X"
- [ ] A folder has been added to `algorithms/` with a meaningful name corresponding to your method name.
- [ ] The added folder includes these elements:
    - [ ] `metadata.yml` (**required**): A file describing your submission, following the descriptions in (algorithms/feat/metadata.yml). 
    - [ ] `regressor.py` (**required**): a Python file that defines your method, named appropriately. See [algorithms/feat/regressor.py][regressor] for complete documentation. 
      `regressor.py` contains:
      - [ ] `est`: a sklearn-compatible `Regressor` object. 
      - [ ]  `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. 
      - [ ]  `eval_kwargs` *(optional)*: a dictionary that can specify method-specific arguments to `evaluate_model.py`.
    - [ ] `LICENSE` *(optional)* A license file
    - [ ] `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
  It will be used to update the baseline environment (`environment.yml` in the root directory). 
  To the extent possible, conda should be used to specify the dependencies you need. 
  If your method is part of conda, great! You can just put that in here and leave `install.sh` blank. 
    - [ ] `requirements.txt` *(optional)*: a pypi requirements file. The script will run `pip install -r requirements.txt` if this file is found, before proceeding.
    - [ ] `install.sh` *(optional)*: a bash script that installs your method **without sudo permissions**. 
- [ ] I did not include source code; instead I used `install.sh` to pull it from a stable source repository. 

- [ ] I locally tested that `bash local_ci.sh [method-folder-name]` runs successfully without error. 