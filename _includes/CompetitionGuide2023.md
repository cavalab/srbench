# Symbolic Regression GECCO Competition - 2023

tl;dr the second edition of the symbolic regression competition will allow the participants to experiment with their own pipeline to submit the best symbolic regression model. We will have two tracks: best accuracy-simplicity trade-off and best pre and post-analysis. Read below for more details and don’t forget to join our Discord server at [https://discord.gg/Dahqh3Chwy](https://discord.gg/Dahqh3Chwy).

The participation link is now online! See [Entries](#entries).

# Results

In total there were 10 official participants for Track 1 and 4 for Track 2. We will contact the participants soon regarding the prize and a power session to get some feedback of the competition. Also, we will write a technical report about the results to be published by next year. 

**Track 1 final ranks:**

| Team                     | Participants                                              | method                           | score    | rank | src | link |
|--------------------------|-----------------------------------------------------------|----------------------------------|----------|------|-----|-----|
| pksm                     | Parshin Shojaee Kazem Meidan                              | TPSR                             | 6.307885 | 1    | N   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-pksm |
| newton-sr                | Nicolas Lassabe Paul Gersberg                             | NewTonSR++                       | 6.224784 | 2    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-newton-sr |
| sarma                    | Aleksandar Kartelj Marko Djukanovic                       | RILS                             | 6.136364 | 3    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-sarma |
| player                   | Lianjie Zhong Jinghui Zhong Dongjunlan Nikola Gligorovski | PFGP                             | 5.448649 | 4    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-player |
| stackgp                  | Nathan Haut                                               | stackgp                          | 5.130641 | 5    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-stackgp |
| university-of-wellington | Hengzhe Zhang  Qi Chen  Bing Xue  Mengjie Zhang           | SR-Forest                        | 4.251969 | 6    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-victoria-university-of-wellington |
| wonderful-time           | Hai Minh Nguyen                                           | SymMFEA                          | 3.440273 | 7    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-wonderful-time |
| his_jsr                  | Gurushant Gurushant  Jatinkumar Nakrani Rajni Maandi      | LR + gplearn                     | 3.43949  | 8    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-his_jsr_2023 |
| tontakt                  | Andrzej Odrzywołek                                        | enumeration, PySR, rational poly | 2.855524 | 9    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-tontakt |
| amir                     | Mohammad Amirul Islam                                     | PySR                             | 1.788926 | 10   | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-1-amir |

**Track 2 final ranks:**

| Team                     | Participants                                                | method    | score | rank | src | link |
|--------------------------|-------------------------------------------------------------|-----------|-------|------|-----|-----|
| university-of-wellington | Hengzhe Zhang  Qi Chen  Bing Xue  Mengjie Zhang             | SR-Forest | 3.25  | 1    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-2-victoria-university-of-wellington |
| player                   | Lianjie Zhong Jinghui Zhong Dongjunlan Nikola Gligorovski   | PFGP      | 2.83  | 2    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-2-player |
| his_jsr                  | Gurushant Gurushant Jatinkumar Nakrani Rajni Maandi         | gplearn   | 2.25  | 3    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-2-his_jsr_2023 |
| c-bio-ufpr               | Adriel Macena Falcão Martins  Aurora Trinidad Ramirez Pozo  | PySR      | 1.75  | 4    | Y   | https://github.com/ufabc-bcc/srbench-competition-2023-track-2-c-bio-ufpr |

All of the competition source-codes, datasets, and results are also available as a single zip file at https://doi.org/10.5281/zenodo.8283005


# Call for participation

The 2023 edition of the Symbolic Regression (SR) competition will be composed of two tracks: performance track and interpretability track. 
The participants will have the freedom to apply their own pipeline with the objective of returning symbolic models that best describe the data. 
In the first track, the models will be evaluated according to accuracy and simplicity. 
In the second track, participants are further asked to provide a post-analysis focused on the interpretation of their symbolic model.

The competition will start 01-April-2023 and last 2 months. The participants will gain access to 3 synthetic data sets for the first track, and 1 real-world data set for the second track. The tracks run independently and participants can enroll in one or both of them.
During the two months, the participants can apply an SR approach or pipeline of their choice, e.g., their own novel algorithm or an existing SR package, as well as pre- and post-processing methods (e.g., feature construction and model simplification, respectively) to find suitable symbolic models for the corresponding data sets. 

Enrollment will be done via GitHub Classroom, at a link to be announced soon. 
This will result in a private repository that can be accessed only by the participating team and the organizers.  
The repository contains detailed submission instructions, a default directory structure, and the data sets for the two tracks.

# Track 1 - Performance

The participants will be free to experiment with these data sets until the deadline. 
Analysis on each dataset should include the production of a single best model for each dataset, and an extended abstract discussing the pipeline.

At the competition submission deadline the repository of a participating team should contain:

- [required] A file containing a single model as a **sympy-compatible expression**, selected as the best expression for that data set, named `dataset_X_best_model`.
- [required] A maximum 4 page extended abstract in PDF format describing the algorithm and pipeline. This PDF must contain the name and contact information of the participants.
- [to be eligible for prize] Reproducibility documentation in the `src` folder.
    - Installation and execution instructions 
    - Scripts (Jupyter Notebook is accepted) with detailed instructions of all steps used to produce models (including hyperparameters search, if any) 

## Evaluation criteria

The final score of each competitor will be composed of:

- *acc*: Rank based on the accuracy (R^2) on a separate validation set for each data set.
- *simpl*: Rank based on the a simplicity (number of nodes calculating by traversing the sympy expression) of the model for each data set.

The rank will be calculated for each data set independently such that, with N participants, the k-th ranked competitor (k=1 being the best) will be assigned a value of *N - k + 1*. The final score will be the harmonic mean of all of the scores and each participant will be ranked accordingly:

```python
score = 2*n / sum([ (1/acc[i]) + (1/simpl[i]) for i in (1..n)])
```

# Track 2 - Interpretability

The participants will be free to experiment with this data set until the deadline. 
Analysis on each dataset should include the production of of one or more models and a detailed pre and post-analysis regarding the interpretation of that model.
Together with the data set we will also provide the context (what the data is about, how it was extracted, etc.) and a description of each feature.
The interpretability analysis in the extended abstract can contain any information that can be extracted from the symbolic expressions. 
For example, you can try to analyze the behavior of the target value w.r.t. certain features, make a study of how some features interact, measure the uncertainty of the predictions or confidence intervals, and explain whether these results are reasonable given the nature of the data. 
Extra points will be awarded for analysis that is unique to Symbolic Regression models.

At the competition submission deadline the repository of a participating team should contain:

- [required] The result files in the `result` folder. A file called `dataset_best_models` containing relevant models as a **sympy-compatible expression**, selected as the best expression for that data set.
- [required] A 4 page extended abstract in PDF format describing the algorithm, pipeline, and the intepretability analysis of the real-world data set (`paper` folder). This PDF must contain the name and contact information of the participants.
- [to be eligible for prize] Reproducibility documentation in the `src` folder.
    - Installation and execution instructions 
    - Scripts (Jupyter Notebook is accepted) with detailed instructions of all steps used to produce models (including hyperparameters search, if any) 
    - Code or instruction to compute any additonal analysis, if applicable.

## Evaluation criteria

Each member of the jury will assign a score to each submission and the final score will be a simple average of the assigned scores. The jury will take into consideration:

- Level of details in the pipeline
- Readability of the model
- Interestingness of the pre and post analysis process
- Analysis of interpretation (with special points for analysis that can only be made using SR models)

Notice that the scores are subjective and these criteria are only a guideline to the jury (e.g., participants who provide a large model with a very good interpretation may score more points than participants that provide a small model with a less good interpretation).

# Prize money

As to stimulate reproducibility, a prize money of US$ 750.00 will be granted to the best ranked open-sourced participant (those with a complete `src` folder) **for each track**.
For both tracks, we will try to organize a special issue for the participants who are interested in publishing their methods and results. 

**Important:** the participants must ensure that the evaluation (prediction) of the models obtained from their pipeline is the same as those of the respective sympy-compatible expressions.

**Important (2):** all of the repositories will be made public after the competition ends.

# Come chat with us!!

If you want to ask any question, provide some feedback or simply chit-chat, join us at the Discord server: [https://discord.gg/Dahqh3Chwy](https://discord.gg/Dahqh3Chwy)

# Entries

To participate of the competition, each member of the team must click on the links corresponding to the desired track. The first member to enroll will create a team name and the other members will choose their team from the list of choices. After enrolling, Github Classroom will set up a private github repository to be used by each team. Read the instructions inside the repository carefully and that's it, you are good to go!

Track 1: [https://classroom.github.com/a/2QvkhUcx](https://classroom.github.com/a/2QvkhUcx)

Track 2: [https://classroom.github.com/a/B9mQ5SlP](https://classroom.github.com/a/B9mQ5SlP)


# Organizers

Fabricio Olivetti de França - Federal University of ABC - folivetti (at) ufabc.edu.br

Fabricio Olivetti de França is an associated professor in the Center for Mathematics, Computing and Cognition (CMCC) at Federal University of ABC. He received his PhD in Computer and Electrical Engineering from State University of Campinas. His current research topics are Symbolic Regression, Evolutionary Computation and Functional Data Structures.

Marco Virgolin - Centrum Wiskunde & Informatica - marco.virgolin (at) cwi.nl

Marco Virgolin is a researcher (tenure track) at Centrum Wiskunde & Informatica (CWI), the Dutch national center of mathematics and computer science. He received his PhD from Delft University of Technology. Marco works on explainable AI, most notably by means of evolutionary machine learning methods such as genetic programming. He is also interested in medical applications of machine learning and human-machine interaction.

Pierre-Alexandre Kamienny - Meta AI & Sorbonne University

Pierre-Alexandre is a third-year PhD student at Meta AI Paris and Sorbonne University. He worked on fast adaptation in deep reinforcement learning and has been focused for more than a year on developing neural methods for symbolic regression that exhibits properties such as leveraging past experience as well as fast inference.

Geoffrey Bomarito - NASA Langley Research Center - geoffrey.f.bomarito (at) nasa.gov

Geoffrey is a research engineer at NASA Langley Research Center in the USA. His research centers on uncertainty quantification in machine learning with a goal of building trust in data-driven models. Specifically, his recent work focuses on genetic programming based symbolic regression, physics-informed generative adversarial networks, and multifidelity uncertainty quantification.

# Sponsors

<a href="http://www.chip.org" ><img style="float:center;" src="../assets/images/chip-logo_0.png"></a>
<a href="http://www.chip.org" ><img style="float:center; height:60px;" src="../assets/images/bch-hvd.png"></a>
<br>
<a href="https://heal.heuristiclab.com" ><img style="float:center;height:60px;" src="../assets/images/20211004_HEAL-Logo_v7.png"></a>
<a href="https://heal.heuristiclab.com" ><img style="float:center;height:60px;" src="../assets/images/uasau.png"></a>
