---
title: Symbolic Regression Benchmarks
---

{% capture notice-2 %}
# Special Announcement: 2022 SRBench Competition 

We are pleased to announce the first [**SRBench Competition: Interpretable Symbolic Regression for Data Science**](/competition-2022/) which will be hosted at GECCO 2020 in Boston, MA (and online). 
Deadline for entry is May 1, 2022; see the competition page for more information and stay tuned as details are announced. 
{% endcapture %}

<div class="notice--info">{{ notice-2 | markdownify }}</div>

---

The methods for symbolic regression (SR) have come a long way since the days of Koza-style genetic programming (GP).
Our goal with this project is to keep a living benchmark of modern symbolic regression, in the context of state-of-the-art ML methods.

Currently these are the challenges, as we see it:

- Lack of cross-pollination between the GP community and the ML community (different conferences, journals, societies etc)
- Lack of strong benchmarks in SR literature (small problems, toy datasets, weak comparator methods)
- Lack of a unified framework for SR, or GP

We are addressing the lack of pollination by making these comparisons open source, reproduceable and public, and hoping to share them widely with the entire ML research community.
We are trying to address the lack of strong benchmarks by providing open source benchmarking of many SR methods on large sets of problems, with strong baselines for comparison. 
To handle the lack of a unified framework, we've specified minimal requirements for contributing a method to this benchmark: a scikit-learn compatible API.

# References

[v2.0](https://github.com/EpistasisLab/srbench/releases/tag/v2.0) of the benchmark is described in a Neurips 2021 paper:

La Cava, W., Orzechowski, P., Burlacu, B., França, F. O. de, Virgolin, M., Jin, Y., Kommenda, M., & Moore, J. H. (2021). 
Contemporary Symbolic Regression Methods and their Relative Performance. 
*Neurips Track on Datasets and Benchmarks*.
[PMLR](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c0c7c76d30bd3dcaefc96f40275bdc0a-Abstract-round1.html)
, [arXiv](https://arxiv.org/abs/2107.14351)


[v1.0](https://github.com/EpistasisLab/srbench/releases/tag/v1.0) was reported in a GECCO 2018 paper: 

Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. 
[ACM](https://doi.org/10.1145/3205455.3205539)
, [arXiv](http://arxiv.org/abs/1804.09331)


# Benchmarked Methods

This benchmark currently consists of **14** symbolic regression methods, **7** other ML methods, and **252** datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks), including real-world and synthetic datasets from processes with and without ground-truth models.

Methods currently benchmarked:

- Age-Fitness Pareto Optimization (Schmidt and Lipson 2009) 
    [paper](https://dl.acm.org/doi/pdf/10.1145/1830483.1830584)
    , 
    [code](https://github.com/EpistasisLab/ellyn)
- Age-Fitness Pareto Optimization with Co-evolved Fitness Predictors (Schmidt and Lipson 2009) 
    [paper](https://dl.acm.org/doi/pdf/10.1145/1830483.1830584?casa_token=8fAFUrPlfuUAAAAA:u0QJvX-cC8rPtdZri-Jd4ZxcnRSIF_Fu2Vn5n-oXVNu_i71J6ZECx28ucLPOLQY628drsEbg4aFvTw)
    , 
    [code](https://github.com/EpistasisLab/ellyn)
- AIFeynman 2.0 (Udrescu et al. 2020)
    [paper](https://arxiv.org/abs/2006.10782)
    ,
    [code](https://github.com/SJ001/AI-Feynman)
- Bayesian Symbolic Regression (Jin et al. 2020)
    [paper](https://arxiv.org/abs/1910.08892)
    ,
    [code](https://github.com/EpistasisLab/ellyn)
- Deep Symbolic Regression (Petersen et al. 2020)
    [paper](https://arxiv.org/pdf/1912.04871)
    , 
    [code](https://github.com/EpistasisLab/ellyn)
- Fast Function Extraction (McConaghy 2011)
    [paper](http://trent.st/content/2011-GPTP-FFX-paper.pdf)
    ,
    [code](https://github.com/EpistasisLab/ellyn)
- Feature Engineering Automation Tool (La Cava et al. 2017)
    [paper](https://arxiv.org/abs/1807.00981)
    ,
    [code](https://github.com/EpistasisLab/ellyn)
- epsilon-Lexicase Selection (La Cava et al. 2016)
    [paper](https://arxiv.org/abs/1905.13266)
    ,
    [code](https://github.com/EpistasisLab/ellyn)
- GP-based Gene-pool Optimal Mixing Evolutionary Algorithm (Virgolin et al. 2017)
    [paper](https://dl.acm.org/doi/pdf/10.1145/3071178.3071287?casa_token=CHa8EK_ic5gAAAAA:mOAOCu6CL-jHobGWKD2wco4NbpCyS-XTY5thb1dPPsyUkTkLHzmLMF41MWMGWLyFv1G8n-VFaqmXSw)
    ,
    [code](https://github.com/marcovirgolin/GP-GOMEA/)
- gplearn (Stephens)
    [code](https://github.com/trevorstephens/gplearn)
- Interaction-Transformation Evolutionary Algorithm (de Franca and Aldeia, 2020)
    [paper](https://www.mitpressjournals.org/doi/abs/10.1162/evco_a_00285)
    ,
    [code](https://github.com/folivetti/ITEA/)
- Multiple Regression GP (Arnaldo et al. 2014)
    [paper](https://dl.acm.org/doi/pdf/10.1145/2576768.2598291?casa_token=Oh2e7jDBgl0AAAAA:YmYJhFniOrU0yIhsqrHGzUN_60veH56tfwizre94uImDpYyp9RcadUyv_VZf8gH7v3uo5SxjjIPPUA)
    ,
    [code](https://github.com/flexgp/gp-learners)
- Operon (Burcalu et al. 2020)
    [paper](https://dl.acm.org/doi/pdf/10.1145/3377929.3398099?casa_token=HJgFp342K0sAAAAA:3Xbelm-5YjcIgjMvqLcyoTYdB0wNR0S4bYcQBGUiwOuwqbFfV6YnE8YKGINija_V6wCi6dahvQ3Pxg)
    ,
    [code](https://github.com/heal-research/operon)
- Semantic Backpropagation GP (Virgolin et al. 2019)
    [paper](https://dl.acm.org/doi/pdf/10.1145/3321707.3321758?casa_token=v43VobsGalkAAAAA:Vj8S9mHAv-H4tLm_GCL4DJdfW3e5SVUtD6J3gIQh0vrNzM3s6psjl-bwO2NMnxLN0thRJ561OZ0sQA)
    ,
    [code](https://github.com/marcovirgolin/GP-GOMEA)

# Contribute

We are actively updating and expanding this benchmark. 
Want to add your method? 
See our [Contribution Guide.](https://github.com/EpistasisLab/srbench/blob/master/CONTRIBUTING.md)

# How to run

## Installation

We have provided a [conda environment](https://github.com/EpistasisLab/srbench/blob/master/environment.yml), [configuration script](https://github.com/EpistasisLab/srbench/blob/master/configure.sh) and [installation script](https://github.com/EpistasisLab/srbench/blob/master/install.sh) that should make installation straightforward.
We've currently tested this on Ubuntu and CentOS. 
Steps:

1. Install the conda environment:

```bash
conda env create -f environment.yml
conda activate srbench
```

2. Install the benchmark methods:

```bash
bash install.sh
```

3. Checkout the feynman PMLB branch (once these new datasets are merged, you will be able to skip this step):

```bash
git clone https://github.com/EpistasisLab/pmlb/tree/feynman [/path/to/pmlb/]
cd /path/to/pmlb
git lfs fetch
```

## Start the benchmark

Experiments are launched from the `experiments/` folder via the script `analyze.py`.
The script can be configured to run the experiment in parallel locally, on an LSF job scheduler, or on a SLURM job scheduler. 
To see the full set of options, run `python analyze.py -h`. 

After installing and configuring the conda environment, the complete black-box experiment can be started via the command:

```bash
python analyze.py /path/to/pmlb/datasets -n_trials 10 -results ../results -time_limit 48:00
```

Similarly, the ground-truth regression experiment for strogatz datasets and a target noise of 0.0 are run by the command:

```bash
python analyze.py -results ../results_sym_data -target_noise 0.0 "/path/to/pmlb/datasets/strogatz*" -sym_data -n_trials 10 -time_limit 9:00 -tuned
```

# Contact

William La Cava ([@lacava](https://github.com/lacava)), william dot lacava at childrens dot harvard dot edu
