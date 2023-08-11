# This example submission shows the submission of FEAT (cavalab.org/feat). 
from feat import FeatRegressor

"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
est = FeatRegressor(
                    pop_size=100,
                    gens=100,
                    max_time=8*60*60,  # 8 hrs
                    max_depth=6,
                    verbosity=0,
                    batch_size=100,
                    functions ='+,-,*,/,^2,^3,sqrt,sin,cos,exp,log',
                    otype='f'
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
    model_str = est.get_eqn()
    model_str = model_str.replace('|','')

    # use python syntax for exponents
    model_str = model_str.replace('^','**')

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

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    if est.batch_size < len(X):
        est.gens = int(est.gens*len(X)/est.batch_size)
    print('FEAT gens adjusted to',est.gens)
    # adjust max dim
    est.max_dim=min(max(est.max_dim, X.shape[1]), 20)
    print('FEAT max_dim set to',est.max_dim)

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {'gens': 5,
                                  'pop_size': 10
                                 }
                  )
