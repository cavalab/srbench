<!--
Thanks very much for submitting to the SRBench 2022 Competition!

Please check the list below and make sure you have everything covered. 
If you need help with the PR, feel free to tag @srbench-comp and we'll respond asap. 

-->

Competition Checklist:
- [ ] title of this PR is meaningful, i.e. "method X for comp"
- [ ] A folder has been added to `submission/` with a meaningful name corresponding to your method name.
- The added folder includes these elements:
    - [ ] `metadata.yml` (**required**): A file describing your submission, following the descriptions in `example/metadata.yml`.  
    - [ ] `regressor.py` (**required**): a Python file that defines your method, named appropriately. See [submission/feat-example/regressor.py](https://github.com/cavalab/srbench/blob/Competition2022/submission/feat-example/regressor.py) for complete documentation.  It contains:
        - [ ]  `est`: a sklearn-compatible `Regressor` object. 
        - [ ]  `model(est, X=None)`: a function that returns a [**sympy-compatible**](https://www.sympy.org) string specifying the final model. It can optionally take the training data as an input argument. See [guidance below](###-returning-a-sympy-compatible-model-string). 
        - [ ]  `eval_kwargs` (optional): a dictionary that can specify method-specific arguments to `evaluate_model.py`.
        
    - [ ] `LICENSE` *(optional)* A license file
    - [ ] `environment.yml` *(optional)*: a [conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that specifies dependencies for your submission. 
    - [ ] `install.sh` *(optional)*: a bash script that installs your method. 
    - [ ] additional files *(optional)*: you may include a folder containing the code for your method in the submission. 

I have verified that:

- [ ] install scripts do not require sudo permissions. 
- [ ] if pulled remotely, the source code is a fixed version (i.e., rerunning `install.sh` shouldn't pulll a different version of the code when run multiple times.) 


Refer to the [competition guide]( https://cavalab.org/srbench/competition-guide/ ) if you are unsure about any steps. 
If you don't find an answer, ping us!
