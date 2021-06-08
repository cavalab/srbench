# SRBench: A Living Benchmark for Symbolic Regression

This project focuses on benchmarking modern symbolic regression methods in comparison to other common machine learning methods. 
This benchmark currently consists of 14 symbolic regression methods, 7 other ML methods, and 252 datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks), including real-world and synthetic datasets from processes with and without ground-truth models.


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

A preprint of the current version of the benchmark is under review on [Open Review](https://openreview.net/forum?id=xVQMrDLyGst&noteId=4TlmQBkmXvx). 

[v1.0](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v1.0) was reported in our GECCO 2018 paper: 

Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. [DOI](https://doi.org/10.1145/3205455.3205539), [Preprint](https://www.researchgate.net/profile/Patryk_Orzechowski/publication/324769381_Where_are_we_now_A_large_benchmark_study_of_recent_symbolic_regression_methods/links/5ae779b70f7e9b837d392dc9/Where-are-we-now-A-large-benchmark-study-of-recent-symbolic-regression-methods.pdf)


# Contact

William La Cava (@lacava), lacava at upenn dot edu

Patryk Orzechowski (@athril), patryk dot orzechowski at gmail dot com
