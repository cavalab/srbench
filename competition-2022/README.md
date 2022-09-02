# SRBench Competition 2022: Interpretable Symbolic Regression for Data Science

SRBench hosted its first competition at the [GECCO 2022](https://gecco-2022.sigevo.org/) conference in Boston, MA. 
This competition seeks to distill algorithmic design choices and improve the practice of symbolic regression by evaluating the submitted symbolic regression methods on previously unseen, real-world and synthetic datasets. 

This folder contains the code from that competition. 

## Overview

- `experiment/`: contains the main code to run the experiment. 
- `official_competitors/`: contains the submission code from each competitor. 
- `postprocessing/`: contains notebooks to generate comparison figures and stats used throughout the competition. 

All runs were conducted in a linux environment running CentOS7, and most analysis was on Ubuntu 20.04. 

## Installation

``install_competitors.sh`` takes a method in `official_competitors` as an argument, or loops through each and attempts to 

1. make a conda environment for the method
2. install the method

To test the installations, see `test.sh`. 

## Running the code

The script files for each stage are named `submit_stage[X].sh`, where X is stage 0, 1, or 2. 
Each script is calling `analyze.py` with a different configuration. 
For example, Stage 1 of the competition can be run with the command

```
python analyze.py data/stage1/data/*data.csv -results ../results_stage1 -n_jobs 8 -m 16384 -n_trials 1 -stage 1 -time_limit "01:05" 
```

If you don't want to run the code, you can download the results files from zenodo: 

<a href="https://doi.org/10.5281/zenodo.6842176"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6842176.svg" alt="DOI" style="height:20px;" ></a>


