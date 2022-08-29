[judging]: https://github.com/folivetti/srbench-competition-2022/blob/main/evaluation/README.md
# Properties of datasets

We could design several synthetic data sets for each property we intend to measure. Each of these properties contributes a score, preferably normalized between 0 and 1, to the final evaluation (cf. [judging]).

Here's a list of properties we discussed in our latest meeting.

## Re-discovery of exact formula
This may be something that we keep consistent every year. We can, in principle, measure this property on each one of the synthetic data sets we prepare to measure other properties, but treating them as if they were designed for this property.
Maybe, by adding noise to those data sets before testing for this.
The most important question here is how to measure how "close" a model gets(post `sympy.simplify`) to the true formula.

## Escape local optima
Let the function to discover *f* be, e.g., a weighed sum of subfunctions *g_1, g_2, ..., g_k*. Each *g_i*  depends on some "true" features *x_(i,1), x_(i,2), ..., x_(i,d)*. Now, what we do is to further provide some surrogate approximations of the *g*'s as features *z_1, z_2, ..., z_k*, where the surrogate feature *z_i ~ g_i*. Namely, we can set *z_i = g_i(x) + N(\mu_i,\sigma_i)*. For an SR algorithm, it is thus *easier* to discover *f* as a function of the surrogate features instead of also re-discovering the *g*'s. However, using the surrogate features will provide a worse approximation of *f*.

>comment mkommend

It is rather difficult to measure this property directly. A similar approach as for relevant features might be feasible, which is based on calculating a score based on the number of irrelevant surrogate features. However, this property would be strongly correlated with accuracy, because only using the correct features would yield the maximum possible accuracy. 

## Relevant features
For such a task numerous irrelevant features are included within the datasets and the algorithms are scored depending on the number of irrelevant features included. This is somehow similar than the property we test to escape local optima, but should include much more irrelevant features and the data generating function does not necessary consist of a weighted sum of subfunctions. 

m = set of features <BR>
r = set of relevant features <BR>
i = m  \\ r = set of irrelevant features <BR>
ff = set of features is in the model f

*S1 = 1 - | i ∩ f | / | i |* <BR>
*S2 = | r ∩ f | / | f |*

We could either formulate the score in a negative way (S1) so that the score is based on the relative number of avoided irrelevant features => a model using no irrelevant features receives the best score of 1. The relevant features are implicitly judged by the achieved accuracy.

An alternative is that we base our score on the number of correctly identified relevant features (S2). However, this would not penalize models containing several irrelevant features (except in the simplicity / complexity score). 

My choice would be to use S1 for this property, but that is open for discussion. I would not combine both score possibilities as that drastically complicates the score

## Extrapolation
This property is pretty straight forward to test and score. We define the test partition to contain out-of-sample data, or even better define an additional extrapolation partition, where we test the model accuracy on.

The score is then the achieved accuracy on this additional extrapolation partition.

## Sensitivity to noise
The idea behind this property is to test how algorithms are affected by noise in the data. My approach would be to have the same data generating function augmented with different level of noise (e.g 0%, 5%, 10%, 20% of the target's variance => max R² = 1.0, 0.95, 0.9, 0.8, ...). Then we can test how much worse the model quality gets with increasing level of noise. 

The score then be calculated as the ratio of the difference in quality to the difference in noise levels. 

Q1 = quality at noise level 1 (R²) <BR>
Q2 = quality at noise level 2 (R²) <BR>
n1 = noise level 1 <BR>
n2 = noise level 2 <BR>

score = abs(Q1- Q2) / abs(n1 - n2)

Difficulties with such a score arise when bad / inaccurate models have the same quality for different noise levels, e.g. Q1 = Q2 = 0.4, n1 = 0.2, n2 = 0.1 => score = 0. However, this might be caught by the quality metric which is always part of the evaluation. 
