from sklearn.base import RegressorMixin

from pymgpg.sk import MGPGRegressor


est: RegressorMixin = MGPGRegressor(
    verbose=False,
    log=False,
    MO_mode=False,
    n_clusters=1,
    use_optim=True,
    log_front=False,
    drift=True,
    remove_duplicates=True,
    replacement_strategy="sample",
    pop=4096,
    use_adf=True,
    nr_multi_trees=4,
    use_max_range=True,
    ff="lsmse",
    bs_opt=256,
    cmp=1.0,
    max_coeffs=-1,
    max_non_improve=-1,
    equal_p_coeffs=True,
    bs=2048,
    # only return 100 models
    max_models=100,
)


def model(est: MGPGRegressor, X=None) -> str:
    """
    Return a sympy-compatible string of the final model.

    Parameters
    ----------
    est: sklearn regressor
        The fitted model.
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model.

    Notes
    -----

    Ensure that the variable names appearing in the model are identical to
    those in the training data, `X`, which is a `pd.Dataframe`.
    If your method names variables some other way, e.g. `[x_0 ... x_m]`,
    you can specify a mapping in the `model` function such as:

        ```
        def model(est, X):
            mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
            new_model = est.model_
            for k,v in mapping.items():
                new_model = new_model.replace(k,v)
        ```

    If you have special operators such as protected division or protected log,
    you will need to handle these to assure they conform to sympy format.
    One option is to replace them with the unprotected versions. Post an issue
    if you have further questions:
    https://github.com/cavalab/srbench/issues/new/choose
    """

    return str(est.model)


def get_population(est: MGPGRegressor) -> list[RegressorMixin]:
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

    def make_regressor(m):
        e = MGPGRegressor(**est.get_params())
        e.model = m
        e.models = [m]
        return e

    return [make_regressor(m) for m in est.models]


def get_best_solution(est: MGPGRegressor) -> RegressorMixin:
    """
    Return the best solution from the final model.

    Returns
    -------
    A scikit-learn compatible estimator
    """

    return est


"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function. 
    Allows one to configure aspects of the training process.

Options 
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing. 
        for running the tests. called as 
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it. 
        if 0, use all training samples for fit. 
    scale_x: bool, default = True 
        Normalize the input data prior to fit. 
    scale_y: bool, default = True 
        Normalize the input label prior to fit. 
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit. 
        The function signature should be (est, X, y). 
            est: sklearn regressor; the fitted model. 
            X: pd.DataFrame; the training data. 
            y: training labels.
"""


# def my_pre_train_fn(est, X, y):
#     """In this example we adjust FEAT generations based on the size of X
#     versus relative to FEAT's batch size setting.
#     """
#     if est.batch_size < len(X):
#         est.gens = int(est.gens * len(X) / est.batch_size)
#     print("FEAT gens adjusted to", est.gens)
#     # adjust max dim
#     est.max_dim = min(max(est.max_dim, X.shape[1]), 20)
#     print("FEAT max_dim set to", est.max_dim)


# define eval_kwargs.
eval_kwargs = dict(
    test_params=dict(
        g=2,
        max_time=180,
        verbose=True,
    )
)

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("dataset/air.csv")
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

    est.set_params(**eval_kwargs["test_params"])

    est.fit(X, y)
    print("Archive size:", len(get_population(est)))
