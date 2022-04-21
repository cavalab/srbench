from sympy.printing.printer import Printer
import feyn


class QLatticeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 kind='regression',
                 stypes=None,
                 n_epochs=10,
                 threads='auto',
                 max_complexity=10,
                 query_string=None,
                 loss_function=None,
                 criterion='bic',
                 sample_weights=None,
                 function_names=None,
                 starting_models=None,
                 random_state=None
                 ):
        self.kind = kind
        self.stypes = stypes
        self.n_epochs = n_epochs
        self.threads = threads
        self.max_complexity = max_complexity
        self.query_string = query_string
        self.loss_function = loss_function
        self.criterion = criterion
        self.sample_weights = sample_weights
        self.function_names = function_names
        self.starting_models = starting_models
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):

        input_names, output_name = feature_names(X, y)

        X, y = check_X_y(X, y, y_numeric=True)
        X, y = numpy_check_to_DataFrame(X, y)
        data = pd.concat([X, y], axis=1)

        if input_names is not None:
            data = data.rename(columns=dict(zip(X.columns, list(input_names))))
        if output_name is not None:
            data = data.rename(columns={y.name: output_name})

        data.columns = data.columns.astype(str)

        if self.random_state is None:
            rseed = -1
        else:
            rseed = self.random_state

        if sample_weight is None:
            sample_weight = self.sample_weights
        else:
            sample_weight = _check_sample_weight(sample_weight, X)

        ql = feyn.connect_qlattice()
        ql.reset(rseed)
        self.models_ = ql.auto_run(
            data=data,
            output_name=data.columns[-1],
            kind=self.kind,
            stypes=self.stypes,
            n_epochs=self.n_epochs,
            threads=self.threads,
            max_complexity=self.max_complexity,
            query_string=self.query_string,
            loss_function=self.loss_function,
            criterion=self.criterion,
            sample_weights=sample_weight,
            function_names=self.function_names,
            starting_models=self.starting_models
        )
        return self

    def predict(self, X, n=0):

        check_is_fitted(self)

        input_names = feature_names(X)
        X = check_array(X)

        X = numpy_check_to_DataFrame(X)
        if input_names is not None:
            X = X.rename(columns=dict(zip(X.columns, list(input_names))))
        X.columns = X.columns.astype(str)
        return self.models_[n].predict(X)

    def score(self, X, y, n=0):
        input_names, output_name = feature_names(X, y)
        X, y = check_X_y(X, y, y_numeric=True)
        X, y = numpy_check_to_DataFrame(X, y)
        data = pd.concat([X, y], axis=1)

        if input_names is not None:
            data = data.rename(columns=dict(zip(X.columns, list(input_names))))
        if output_name is not None:
            data = data.rename(columns={y.name: output_name})

        data.columns = data.columns.astype(str)

        return self.models_[n].r2_score(data)


est = QLatticeRegressor(
                        kind='regression',
                        n_epochs=100,
                        max_complexity=10,
                        criterion='wide_parsimony',
                   )
# want to tune your estimator? wrap it in a sklearn CV class.

#do we need to make sure features have the same name as in original data?
def model(est):
    printer = Printer()
    string_model = printer.doprint(est.models_[0].sympify())

    return string_model

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
