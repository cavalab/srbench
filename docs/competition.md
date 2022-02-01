---
layout: page
title: Competition 2022
permalink: /competition-2022/
---

* This will become a table of contents (this text will be scrapped).
{:toc}

# Call for Participation: Interpretable Symbolic Regression for Data Science

SRBench will host its first competition at [GECCO 2022](https://gecco-2022.sigevo.org/) conference in Boston, MA. 
Symbolic regression methods have made tremendous advances in the past decade, and have recently gained interest as the broader scientific community has recognized the importance of interpretable machine learning. 
Despite this, there is little agreement in the field about which algorithms are “state-of-the-art”, and how to best design symbolic regression methods for use in the real world. 
This competition seeks to distill algorithmic design choices and improve the practice of symbolic regression by evaluating the submitted symbolic regression methods on previously unseen, real-world and synthetic datasets. 
These datasets will be sourced mainly from the domains of physics, epidemiology and bioinformatics.

Participants are asked to adapt and submit their symbolic regression algorithms to [SRBench](https://github.com/cavalab/srbench), following the [contributing guidelines](https://github.com/cavalab/srbench/blob/master/CONTRIBUTING.md). 
SRBench will automatically test these methods for conformance with the competition.

After the submission deadline, methods will be tested on previously unseen datasets. 
These datasets cover synthetic and real-world problems, and in each case, either an exact model or a human-designed model will be used for comparison. 
Notice that these will be new benchmarks specifically curated for this competition. 
The current version of SRBench will serve as a first-pass filter for candidate entries. 
As such, participants are free to test and fine-tune their algorithms on the current version of SRBench. 
Algorithm submissions will be judged by their ability to discover the ground-truth models, or, in the case of real-world data, approximate or outperform the expert models with similar or lower complexity.
Winners will be determined based on the accuracy and simplicity of the generated models, both individually and in the Pareto efficient sense. 
After competition, the submitted methods, evaluation procedure, and new datasets will be made publicly available.


## Deadline

Entrants should have their methods submitted by **May 1, 2022**. 

## Previous History

This is a new competition, but is based upon the recent work of several symbolic regression researchers to build a comprehensive benchmark framework called [SRBench](cavalab.org/srbench) [^1].
SRBench already benchmarked 21 methods on 252 regression problems, including real-world and synthetic problems from several domains. 
In addition, the repository includes a continuous integration environment that is able to handle new submissions. 
We will leverage this aspect of SRBench to gather submissions for the competition, and automatically test them for conformance with the competition analysis pipeline.

[^1]: La Cava, W., Orzechowski, P., Burlacu, B., de França, F. O., Virgolin, M., Jin, Y., Kommenda, M., & Moore, J. H. (2021). Contemporary Symbolic Regression Methods and their Relative Performance. _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_. [arXiv](https://arxiv.org/abs/2107.14351)


## Judging

Exact ranking criteria will be released soon. 

## Prizes

Cash prizes will be awarded to the winners. 
Exact amounts TBD. 
In addition, winners will be invited to speak at the [Symbolic Regression Workshop](https://gecco-2022.sigevo.org/Workshops#SymReg) during the conference. 

## Who can Participate

Anyone! 
Unlike srbench, **we accept open-source _and_ closed-source entries to this competition**. 
Methods must have a sklearn-like interface, as described below.

## How to Participate

We plan to release a tutorial video demonstrating how to enter the competition; stay tuned. 
We will provide a updates to the contributing guide as details solidify.
We will invite submissions to provide 2-page papers, and we have arranged with the Symbolic Regression workshop organizers to invite winners to present at GECCO 2022 as invited speakers.

If you have a CLI method, [see this gist for making it compatible with the competition format (sklearn Regressors).](https://gist.github.com/folivetti/609bc9b854c51968ef90aa675ccaa60d)

In short, the participants must provide via Pull Request (see [Contributing](contributing)), a modification of the environment file to ensure the installation of any required library, an installation script for their open-source method (see examples in repository), a python script setting the variables est, and hyper_params, and the functions complexity, and model. 
_This script must be named MethodRegressorCompetition2022.py (e.g., GSGPRegressorCompetition2022.py)_. 
After submitting, Github will automatically validate the submission and return an error message in case of any problem. 
For any assistance, [open an issue in the srbench repository](http://github.com/cavalab/srbench/issues).

## Dissemination

Participants retain copyright of their entries. 
The results of the competition, including comparison code and full results, will be made available through SRBench.
A summary of the competition will be published, and participants will be invited to co-author. 

## Contact

Please address questions to william dot lacava at childrens dot harvard dot edu. 

- Michael Kommenda
    - University of Applied Sciences Upper Austria
- William La Cava
    - Boston Children’s Hospital and Harvard Medical School
- Maimuna Majumder
    - Boston Children’s Hospital and Harvard Medical School
- Fabricio Olivetti de França
    - Federal University of ABC
- Marco Virgolin
    - Centrum Wiskunde & Informatica
