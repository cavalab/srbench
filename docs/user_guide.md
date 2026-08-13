# User Guide

**NOTE**: if you are trying to reproduce the results from the 2022 Neurips Benchmark paper, you 
should check out the [v2.0 release](https://github.com/cavalab/srbench/releases/tag/v2.0) version of this repo.  

## Installation

### Local install

We have provided a [conda environment](../base_environment.yml), [configuration script](../configure.sh), and [installation script](../scripts/install_algorithm.sh) that should make installation straightforward.
The installation script is the same used internally when building the docker images.

We've currently tested this on Ubuntu and CentOS. 
Steps:

0. (Optional) Using libmamba as the solver to speedup the installation process:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

1. Install the conda environment naming it `srbench`:

```bash
conda env create -f base_environment.yml -n srbench
conda activate srbench
```

2. Install a benchmark algorithm:

The `scripts/install_algorithm.sh` script installs a single algorithm by name. For example, to install gplearn:

```bash
bash scripts/install_algorithm.sh gplearn
```

Replace `gplearn` with the name of any algorithm directory in `algorithms/` to install it into the srbench conda environment.

3. Download the PMLB datasets:

```bash
git clone https://github.com/EpistasisLab/pmlb/ [/path/to/pmlb/]
cd /path/to/pmlb
git lfs pull
```

### Docker install

For Docker users, you can build the algorithm images using docker-compose:

```bash
bash scripts/make_docker_compose_file.sh
docker compose up
```

This generates a `docker-compose.yml` file and pulls all algorithm images locally. To build images locally see [instructions for docker users](#for-docker-users).

To build a specific algorithm image:

```bash
docker compose build feat
```

To run a command in a container:

```bash
docker compose run feat bash test.sh
```

## Reproducing the benchmark results

**NOTE**: these instructions are for the the [v2.0 release](https://github.com/cavalab/srbench/releases/tag/v2.0) version of this repo.  

Experiments are launched from the `experiments/` folder via the script `analyze.py`.
The script can be configured to run the experiment in parallel locally, on an LSF job scheduler, or on a SLURM job scheduler. 
To see the full set of options, run `python analyze.py -h`. 

**WARNING**: running some of the commands below will submit tens of thousands of experiments. 
Use accordingly. 

### Black-box experiment
After installing and configuring the conda environment, the complete black-box experiment can be started via the command:

```bash
python analyze.py /path/to/pmlb/datasets -n_trials 10 -results ../results_blackbox -time_limit 48:00
```

### Ground-truth experiment

**Train the models**: we train the models subject to varying levels of noise using the options below. 

```bash
# submit the ground-truth dataset experiment. 

for data in "/path/to/pmlb/datasets/strogatz_" "/path/to/pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            $data"*" \ #data folder
            -results ../results_sym_data \ # where the results will be saved
            -target_noise $TN \ # level of noise to add
            -sym_data \ # for datasets with symbolic models
            -n_trials 10 \
            -m 16384 \ # memory limit in MB
            -time_limit 9:00 \ # time limit in hrs
            -job_limit 100000 \ # this will restrict how many jobs actually get submitted.
            -tuned # use the tuned version of the estimators, rather than performing hyperparameter tuning.
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done
```

**Symbolic Assessment**: Following model training, the trained models are assessed for symbolic equivalence with the ground-truth data-generating processes. 
This is handled in [assess_symbolic_model.py](experiment/assess_symbolic_model.py). 
Use `analyze.py` to generate batch calls to this function as follows:

```bash
# assess the ground-truth models that were produced using sympy
for data in "/path/to/pmlb/datasets/strogatz_" "/path/to/pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            -script assess_symbolic_model \
            $data"*" \ #data folder
            -results ../results_sym_data \ # where the results will be saved
            -target_noise $TN \ # level of noise to add
            -sym_data \ # for datasets with symbolic models
            -n_trials 10 \
            -m 8192 \ # memory limit in MB
            -time_limit 1:00 \ # time limit in hrs
            -job_limit 100000 \ # this will restrict how many jobs actually get submitted.
            -tuned # use the tuned version of the estimators, rather than performing hyperparameter tuning.
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done
```

**Output**: next to each `.json` file, an additional file named `.json.updated` is saved with the symbolic assessment included. 

### For docker users

When a new algorithm is submitted to SRBench, a GitHub workflow will generate a docker image and push it to [Docker Hub](hub.docker.com). Ths means that you can also easily pull the images, without having to deal with local installations.

To build the docker images locally, first run `bash scripts/make_docker_compose_file.sh` in the root directory to create a `docker-compose.yml` file describing all images that will be created.
Then `docker compose up` should create all the images.
Instead of creating all images, you can build the image of a specific algorithm with the name of the service (_e.g._ `docker compose build feat`). 

Note: `docker compose up` pulls images from Docker Hub, while `docker compose build` builds them locally.

The file `alg-Dockerfile` specifies the steps used to install the algorithm - you can check it out to see how it will be installed. 
The build relies on `scripts/install_algorithm.sh`, so do not delete this file.

### Post-processing

Navigate to the [postprocessing](postprocessing) folder to begin postprocessing the experiment results. 
The following two scripts collate the `.json` files into two `.feather` files to share results more easily. 
You will notice these `.feather` files are loaded to generate figures in the notebooks. 
They also perform some cleanup like shortening algorithm names, etc.

```
python collate_blackbox_results.py
python collate_groundtruth_results.py
```

**Visualization**

- [groundtruth_results.ipynb](postprocessing/groundtruth_results.ipynb): ground-truth results comparisons
- [blackbox_results.ipynb](postprocessing/blackbox_results.ipynb): ground-truth results comparisons
- [statistical_comparisons.ipynb](postprocessing/statistical_comparisons.ipynb): post-hoc statistical comparisons
- [pmlb_plots](postprocessing/pmlb_plots.ipynb): the [PMLB](https://github.com/EpistasisLab/pmlb) datasets visualization 


## Using your own datasets

To use your own datasets, you want to check out / modify read_file in read_file.py: https://github.com/cavalab/srbench/blob/4cc90adc9c450dad3cb3f82c93136bc2cb3b1a0a/experiment/read_file.py

If your datasets follow the convention of https://github.com/EpistasisLab/pmlb/tree/master/datasets, i.e. they are in a pandas DataFrame with the target column labelled "target", you can call `read_file` directly just passing the filename like you would with any of the PMLB datasets. 
The file should be stored and compressed as a `.tsv.gz` file. 
