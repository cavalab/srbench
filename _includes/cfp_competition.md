<style>
img {
    height: 100px;
    margin: 5px;
}
</style>

<a href="http://www.chip.org" ><img style="float:center;" src="../assets/images/chip-logo_0.png"></a>
<a href="http://www.chip.org" ><img style="float:center; height:60px;" src="../assets/images/bch-hvd.png"></a>
<br>
<a href="https://heal.heuristiclab.com" ><img style="float:center;" src="../assets/images/20211004_HEAL-Logo_v7.png"></a>
<a href="https://heal.heuristiclab.com" ><img style="float:center;height:60px;" src="../assets/images/uasau.png"></a>

{% include toc %}

# Call for Participation: Interpretable Symbolic Regression for Data Science


SRBench will host its first competition at [GECCO 2022](https://gecco-2022.sigevo.org/) conference in Boston, MA. 
Symbolic regression methods have made tremendous advances in the past decade, and have recently gained interest as the broader scientific community has recognized the importance of interpretable machine learning. 
Despite this, there is little agreement in the field about which algorithms are “state-of-the-art”, and how to best design symbolic regression methods for use in the real world. 
This competition seeks to distill algorithmic design choices and improve the practice of symbolic regression by evaluating the submitted symbolic regression methods on previously unseen, real-world and synthetic datasets. 
These datasets will be sourced mainly from the domains of physics, epidemiology and bioinformatics.

**Participants are asked to adapt and submit their symbolic regression algorithms to SRBench following the [Competition Guide](/srbench/competition-guide/).**
SRBench will automatically test these methods for conformance with the competition.

After the submission deadline, methods will be tested on previously unseen datasets. 
These datasets cover synthetic and real-world problems, and in each case, either an exact model or a human-designed model will be used for comparison. 
Notice that these will be new benchmarks specifically curated for this competition. 
The current version of SRBench will serve as a first-pass filter for candidate entries. 
As such, participants are free to test and fine-tune their algorithms on the current version of SRBench. 
Algorithm submissions will be judged by their ability to discover the ground-truth models, or, in the case of real-world data, approximate or outperform the expert models with similar or lower complexity.
Winners will be determined based on the accuracy and simplicity of the generated models, both individually and in the Pareto efficient sense. 
After competition, the submitted methods, evaluation procedure, and new datasets will be made publicly available.

See the [Competition Guide](/srbench/competition-guide/) for detailed instructions. 

## Important Dates

Entrants should have their methods submitted by ~~May 1, 2022~~ **May 15, 2022**. 

Submissions will be accepted starting March 14, 2022.
Note that submissions are tested automatically and must pass all tests to be a competitor. 
Please budget time for this. 
The winners will be announced at GECCO, July 9-13 in Boston, MA, and online.

## Previous History

This is a new competition, but is based upon the recent work of several symbolic regression researchers to build a comprehensive benchmark framework called [SRBench](cavalab.org/srbench) [^1].
SRBench already benchmarked 21 methods on 252 regression problems, including real-world and synthetic problems from several domains. 
In addition, the repository includes a continuous integration environment that is able to handle new submissions. 
We will leverage this aspect of SRBench to gather submissions for the competition, and automatically test them for conformance with the competition analysis pipeline.

[^1]: La Cava, W., Orzechowski, P., Burlacu, B., de França, F. O., Virgolin, M., Jin, Y., Kommenda, M., & Moore, J. H. (2021). Contemporary Symbolic Regression Methods and their Relative Performance. _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_. [arXiv](https://arxiv.org/abs/2107.14351)


## Judging

Exact ranking criteria will be released soon. 
Roughly, judging will be done in three stages:

**Stage 1**: Submitted methods will be benchmarked on datasets from [PMLB](http://github.com/EpistasisLab/pmlb) as a filter for the subsequent stages. 
Methods will be required to perform at least as well as penalized linear regression across this benchmark in terms of test set R2. 

**Stage 2: Synthetic**: Stage 2 will consist of comparisons on synthetic benchmarks. Metrics for evaluation will include *model complexity* (TBD), *accuracy* (R2), and *solution rates* (fraction of problem instances for which a method generates an exact solution). 

**Stage 3: Real-world**: Methods will be compared on a real-world prediction task and reviewed by domain experts. 


## Prizes

1. A total of **$2500 in cash prizes** will be awarded to winning entries of stage 2 and stage 3 (exact amounts/division TBD).
2. Winners will be invited to speak at the [Symbolic Regression Workshop](https://gecco-2022.sigevo.org/Workshops#SymReg) during the conference. 
3. Additional publication opportunities will be released as they become available.

This competition is sponsored by:

- [Computational Health Informatics Program at Boston Children's Hospital / Harvard Medical School](http://www.chip.org)
- [Heuristic and Evolutionary Algorithm Laboratory at the University of Applied Sciences Upper Austria](https://heal.heuristiclab.com)


## Who can Participate

Anyone! 
Whereas SRBench is completely open-source, **we will accept open-source _and_ closed-source entries to this competition**. 
Note that methods cannot rely on external API calls: they must be completely self-contained. 

## How to Participate 

Detailed instructions are in the [Competition Guide](/srbench/competition-guide/). 
We will provide a updates to the contributing guide as details solidify.
We also plan to release a tutorial video demonstrating how to enter the competition; stay tuned.

Participants must provide files for their method via a Pull Request to the [Competition2022 branch](https://github.com/cavalab/srbench/tree/Competition2022) on SRBench. 
In short, submissions consist of a sklearn-like SR Method and an installation script for their method (see examples in repository), and a python script setting variables . 
After submitting, a CI process will automatically test the submission and return an error message in case of any problem. 
For any assistance, [open an issue in the srbench repository](http://github.com/cavalab/srbench/issues).
Methods must pass CI and code review before the competition submission deadline to be considered.


## Dissemination

Participants retain copyright of their entries. 
The results of the competition, including comparison code and full results, will be made available through SRBench.
A summary of the competition will be published, and participants will be invited to co-author. 

## Organizers

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
