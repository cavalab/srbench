# Benchmarking Symbolic Regression Methods

This project focuses on benchmarking modern symbolic regression methods in comparison to other common machine learning methods. 
This benchmark consists of more than 100 datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks).

[v1.0](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v1.0) was reported in our GECCO 2018 paper: 

Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. [DOI](https://doi.org/10.1145/3205455.3205539), [Preprint](https://www.researchgate.net/profile/Patryk_Orzechowski/publication/324769381_Where_are_we_now_A_large_benchmark_study_of_recent_symbolic_regression_methods/links/5ae779b70f7e9b837d392dc9/Where-are-we-now-A-large-benchmark-study-of-recent-symbolic-regression-methods.pdf)

# Contribute

We are actively updating and expanding this benchmark. 
Want to add your method? 
See our [Contribution Guide.](CONTRIBUTING.md)

# How to run

Batch jobs are controlled via `submit_jobs.py`. 
Run `python submit_jobs.py -h` to see options.

Results of single methods on datasets are generated using `analyze.py`. 
Run `python analyze.py -h` to see options. 


# Installation

We have provided a [conda environment](environment.yml) and [installation script](install.sh) that should make installation straightforward.
We've currently tested this on Ubuntu and CentOS. 




# Contact

William La Cava (@lacava), lacava at upenn dot edu

Patryk Orzechowski (@athril), patryk dot orzechowski at gmail dot com
