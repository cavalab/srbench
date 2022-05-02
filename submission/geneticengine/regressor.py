import geneticengine.off_the_shelf.regressors as gengy_regressors

"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
est = gengy_regressors.GeneticProgrammingRegressor(
        population_size = 50,
        n_elites = 5, 
        n_novelties = 10,
        number_of_generations = 50,
        max_depth = 15,
        favor_less_deep_trees = True,
        hill_climbing = False,
        seed = 123,
        probability_mutation = 0.01,
        probability_crossover = 0.9
                   )
# want to tune your estimator? wrap it in a sklearn CV class. 

def model(est, X=None):
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

    # Here we replace "|" with "" to handle
    # protecte sqrt (expressed as sqrt(|.|)) in FEAT) 
    model_str = est.sympy_compatible_phenotype

    return model_str

def my_pre_train_fn():
    pass

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {}
                  )