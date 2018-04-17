# regression-benchmark
A Large benchmark of regression methods including GP-based regression methods.

Validation scripts were run with the following command:
```python
python3.6 validation-[algorithm_name].py [dataset_name].tsv.gz output-[dataset_name].txt [trial_name]
```

e.g.:

```python
python3.6 validation-MLPRegressor.py dataset.tsv.gz dataset-MLPRegressor-results.txt 0
```

