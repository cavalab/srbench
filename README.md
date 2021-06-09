# SRBench: A Living Benchmark for Symbolic Regression

The methods for symbolic regression (SR) have come a long way since the days of Koza-style genetic programming (GP).
Our goal with this project is to keep a living benchmark of modern symbolic regression, in the context of state-of-the-art ML methods.

Currently these are the challenges, as we see it:

- Lack of cross-pollination between the GP community and the ML community (different conferences, journals, societies etc)
- Lack of strong benchmarks in SR literature (small problems, toy datasets, weak comparator methods)
- Lack of a unified framework for SR, or GP

We are addressing the lack of pollination by making these comparisons open source, reproduceable and public, and hoping to share them widely with the entire ML research community.
We are trying to address the lack of strong benchmarks by providing open source benchmarking of many SR methods on large sets of problems, with strong baselines for comparison. 
To handle the lack of a unified framework, we've specified minimal requirements for contributing a method to this benchmark: a scikit-learn compatible API.

# Results

This benchmark currently consists of **14** symbolic regression methods, **7** other ML methods, and **252** datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks), including real-world and synthetic datasets from processes with and without ground-truth models.

[Browse the Current Results](postprocessing/)

Methods currently benchmarked:

- Age-Fitness Pareto Optimization 
- Age-Fitness Pareto Optimization with Fitness Predictors
- Bayesian Symbolic Regression
- Deep Symbolic Regression
- Fast Function Extraction
- Feature Engineering Automation Tool
- epsilon-Lexicase Selection
- GP-based Gene Optimal Mixing Evolutionary Algorithm
- gplearn
- Interaction-Transformation Evolutionary Algorithm
- Multiple Regression GP
- Operon
- Semantic Backpropagation GP  
- AIFeynman 2.0

# Contribute

We are actively updating and expanding this benchmark. 
Want to add your method? 
See our [Contribution Guide.](CONTRIBUTING.md)

# How to run

## Installation

We have provided a [conda environment](environment.yml), [configuration script](configure.sh) and [installation script](install.sh) that should make installation straightforward.
We've currently tested this on Ubuntu and CentOS. 
Steps:

1. Install the conda environment:

```
conda env create -f environment.yml
conda activate srbench
```

2. Install the benchmark methods:

```
bash install.sh
```

3. Checkout the feynman PMLB branch (once these new datasets are merged, you will be able to skip this step):

```
git clone https://github.com/EpistasisLab/pmlb/tree/feynman [/path/to/pmlb/]
cd /path/to/pmlb
git lfs fetch
```

## Start the benchmark

Experiments are launched from the `experiments/` folder via the script `analyze.py`.
The script can be configured to run the experiment in parallel locally, on an LSF job scheduler, or on a SLURM job scheduler. 
To see the full set of options, run `python analyze.py -h`. 

After installing and configuring the conda environment, the complete black-box experiment can be started via the command:

```
python analyze.py /path/to/pmlb/datasets -n_trials 10 -results ../results -time_limit 48:00
```

Similarly, the ground-truth regression experiment for strogatz datasets and a target noise of 0.0 are run by the command:

```
python analyze.py -results ../results_sym_data -target_noise 0.0 "/path/to/pmlb/datasets/strogatz*" -sym_data -n_trials 10 -time_limit 9:00 -tuned
```

# Cite

[v1.0](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v1.0) was reported in our GECCO 2018 paper: 

Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. [DOI](https://doi.org/10.1145/3205455.3205539), [Preprint](https://www.researchgate.net/profile/Patryk_Orzechowski/publication/324769381_Where_are_we_now_A_large_benchmark_study_of_recent_symbolic_regression_methods/links/5ae779b70f7e9b837d392dc9/Where-are-we-now-A-large-benchmark-study-of-recent-symbolic-regression-methods.pdf)


# Contact

William La Cava (@lacava), lacava at upenn dot edu

Patryk Orzechowski (@athril), patryk dot orzechowski at gmail dot com
