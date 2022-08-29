_DEPRECATED, PLEASE REFER TO_ [../JudgingCriteria.md](../JudgingCriteria.md)

```
All that is written here is just a proposal for internal use and super-subject to changes.
Criticism and attempts to destroy this are very well come because this should be *spot on* before we decide to deploy it!
```

# Evaluation
The competition has two tracks: one on *synthetic data sets* and one on a *real-world* data set, the latter involving an expert of the field.

Here we explain how the evaluation is carried out for these two tracks, starting from the real world data set.

## Real-world track
For the real-world data set, an expert will judge the quality of the competing SR algorithms in terms of the models they produce on the real-world data set.
This assessment carried out by the expert is subjective and based on his/her expertise.

Each SR algorithm will be run `N=10` times upon the training set, producing `N` models.
Each model will be tested on the test set, and the model with best generalization accuracy (namely, `r2_score`) will be considered as representative for the respective generating SR algorithm.

Let `K` be the number of competing SR algorithms.
The expert will rank the so-obtained `K` competing models in terms of his or her *trust* in them. 
Trust is a subjective measure decided by the expert.
We will only ask the expert to take into account the level of `accuracy`, together with the notions of `simplicity` and `soundess`.
The expert is free to interpret these notions.
For example, for two models `m_1` and `m_2`, if `m_1` is more accurate than `m_2` only by a small margin, the expert may decide that they are equivalently accurate; moreover, `m_1` may have a smaller number of components than `m_2`, but the expert may decide that `m_2` is more simple/sound because of the nature of the components in use (e.g., `m_2` uses more intuitively-sensible features), as well as their combinations (e.g., `m_2` does not include `log(tanh(arcsin(.)))`).


The winning SR algorithm is the one whose model is ranked 1st in terms of trust.


## Synthetic track
Evaluation for the synthetic track is performed automatically and in an objective manner, as follows:
1. Independently for each data set, SR algorithms are ranked according to different categories: `rank_accuracy`, `rank_complexity`, `rank_property`. Higher rank = better.
  * Each one of these ranks is obtained by averaging across `K=10` runs. 
  * Accuracy is measured in terms of `r2_score`, rounded to the 3rd decimal (equal r2_scores are assigned the same rank).
  * Complexity is measured either in terms of number of components in the model post simplification via `sympy.simplify`.
  * The *property* is a data set-specific property we intend to measure for the synthetic data set.
  For example, we may generate a data set with a number of irrelevant features, and rank the SR algorithms in terms of how many irrelevant features are correctly absent from their models.
  Another example is proximity to the true generating formula.
  Such sort of calculations is done on simplified models (i.e., post `sympy.simplify`), because an SR algorithm may decide to encode beneficial constants by opportunely combining irrelevant features (e.g., `irrelevant_feature_i/irrelevant_feature_i=1`).
2. Next, a single score is computed for each data set by taking the harmonic mean of the three category ranks just mentioned (i.e., `score_dataset = harmonic_mean(rank_accuracy, rank_complexity, rank_property)`); this promotes SR algorithms that produce models that have a good trade-off between the different categories (e.g., an algorithm that produces most-accurate but very complex models will score worse than an algorithm that produces decently-accurate and not too complex models).
3. The final score is obtained by averaging the data-set specific scores: `final_score = mean_i(score_dataset_i)`.

The algorithm with best final score wins this track of the competition.



## Minimal accuracy
Model complexity is an important aspect in our assessment.
Trivially, `mean(y_train)` is likely to be the most-accurate model of minimal complexity: a single constant parameter suffices to encode this model.
Consequently, we require that models have a minimum level of accuracy (in terms of R^2-score), which we calculate with respect to the most-accurate model for the data set in exam.
Namely, a model `m` must satisfy:

`r2_score(y_test, m(X_test)) / max_i(r2_score(y_test, m_i(X_test))) > 0.5`.

If this condition is not satisfied, then the respective SR algorithm is assigned the worst-possible rank on all three categories (point 1 of the previous section). 
The worst-possible rank is equal to the number of algorithms considered; ties are allowed among models that do not meet the minimal accuracy.
Note that this assessment is done independelty for each one of the `K` repetitions.

_Comment by mkommend_

This condition might be somewhat problematic. What about algorithm / model that severly overfit and thus don't reach the necessary minimal accuracy threshold. Is it really desired to assign the worst quality to all three categories? What about using the penalty only if both training and test accuracy are not within 50% of the best accuracy?

_Resp by marco_
Because we filter on being better than ElasticNet on PMLB, we do not need this anymore. Anyway I set this document as deprecated, wrote a new version in ../JudingCriteria.md