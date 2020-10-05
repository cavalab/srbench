# Benchmarking Symbolic Regression Methods

This project focuses on benchmarking modern symbolic regression methods in comparison to other common machine learning methods. 
This benchmark consists of more than 100 datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks).

[v1.0](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v1.0) was reported in our GECCO 2018 paper: 

Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. [DOI](https://doi.org/10.1145/3205455.3205539), [Preprint](https://www.researchgate.net/profile/Patryk_Orzechowski/publication/324769381_Where_are_we_now_A_large_benchmark_study_of_recent_symbolic_regression_methods/links/5ae779b70f7e9b837d392dc9/Where-are-we-now-A-large-benchmark-study-of-recent-symbolic-regression-methods.pdf)

# How to run

Validation scripts were run with the following command:
```python
python3.6 validation-[algorithm_name].py [dataset_name].tsv.gz output-[dataset_name].txt [trial_name]
```

e.g.:

```python
python3.6 validation-MLPRegressor.py dataset.tsv.gz dataset-MLPRegressor-results.txt 0
```
