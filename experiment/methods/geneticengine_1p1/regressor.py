# This example submission shows the submission of FEAT (cavalab.org/feat).
from geml.regressors import RandomSearchRegressor
from geml.regressors import model
from sklearn.base import RegressorMixin

"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
est: RegressorMixin = RandomSearchRegressor(
    max_time=1,  # 8 hrs. Your algorithm should have this feature
)


def get_population(est) -> list[RegressorMixin]:
    """
    Return the final population of the model. This final population should
    be a list with at most 100 individuals. Each of the individuals must
    be compatible with scikit-learn, so they should have a predict method.

    Also, it is expected that the `model()` function can operate with them,
    so they should have a way of getting a simpy string representation.

    Returns
    -------
    A list of scikit-learn compatible estimators
    """

    return est.get_population()


def get_best_solution(est) -> RegressorMixin:
    """
    Return the best solution from the final model.

    Returns
    -------
    A scikit-learn compatible estimator
    """

    return est.get_best_solution()


# define eval_kwargs.
eval_kwargs = {}
