from .gpzgd import GPZGD

est = GPZGD()

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

    model_str = est.expr_str()
    model_str = model_str.replace("np.","").replace("[:,", "").replace("]", "")

    # use python syntax for exponents
    model_str = model_str.replace('^','**')

    if X is not None:
        mapping = { 'X'+str(i):k for i, k in enumerate(X.columns) }
        for k, v in reversed(mapping.items()):
            model_str = model_str.replace(k,v)

    return model_str

################################################################################
# Optional Settings
################################################################################


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

def pre_train_fn(est, X, y):
    if len(X) <= 1000:
        pop_size = 500
        generations = 5000
        max_time = 3564 ## 99% of one hour
    else:
        pop_size = 200
        generations = 5000
        max_time = 35640 ## 99% of ten hours

    est.tournament_size = 10
    est.pop_size = pop_size
    est.generations = generations
    est.timeout = max_time
    
# define eval_kwargs.
eval_kwargs = {
    "pre_train" : pre_train_fn
}
