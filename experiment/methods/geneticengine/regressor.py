from audioop import cross
from typing import Union
import geneticengine.off_the_shelf.regressors as gengy_regressors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""

class OptimisedGPRegressor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        population_size = 50,
        n_novelties = 10,
        favor_less_deep_trees = True,
        random_state = 123,
        max_time = 100,
        optimisation_dedicated_proportion = 0.2,
        slack_time = 0.05,
    ):
        self.population_size = population_size
        self.n_novelties = n_novelties
        self.favor_less_deep_trees = favor_less_deep_trees
        self.random_state = random_state
        self.max_time = max_time
        self.optimisation_dedicated_proportion = optimisation_dedicated_proportion
        self.slack_time = slack_time
        self.model = None
    

    def fit(self, X, y):
        n_elites = [ 5, 10 ]
        max_depths = [ 10, 15 ]
        hill_climbings = [ True, False ]
        mutation_probs = [ 0.01, 0.05, 0.1, 0.2 ]
        crossover_probs = [ 0.8, 0.9, 0.95 ]
        
        CVS = 2
        
        param_grid_size = len(n_elites) * len(max_depths) * len(hill_climbings) * len(mutation_probs) * len(crossover_probs)
        param_alloted_time = int((self.max_time * self.optimisation_dedicated_proportion) / (param_grid_size * CVS))
        if param_alloted_time < 1: # For testing
            n_elites = [ 5 ]
            max_depths = [ 10 ]
            mutation_probs = [ 0.01 ]
            crossover_probs = [ 0.8 ]

            param_alloted_time = 1
            
        param_grid: Union[dict, list] = { 
                                "population_size": [ self.population_size ],
                                "n_elites": n_elites,
                                "n_novelties": [ self.n_novelties ],
                                "max_depth": max_depths,
                                "favor_less_deep_trees": [ self.favor_less_deep_trees ],
                                "seed": [ self.random_state ],
                                "hill_climbing": hill_climbings,
                                "probability_mutation": mutation_probs,
                                "probability_crossover": crossover_probs,
                                "timer_stop_criteria": [ True ],
                                "timer_limit": [ param_alloted_time ],
                                "metric": [ 'r2' ],
                                }

        grid_search = GridSearchCV(gengy_regressors.GeneticProgrammingRegressor(),param_grid,cv=CVS)
        
        grid_search.fit(X,y)
        model = grid_search.best_estimator_
        
        model_alloted_time = int(self.max_time * (1 - self.optimisation_dedicated_proportion - self.slack_time))
        if "timer_limit" in model.get_params():
            model.set_params(timer_limit=model_alloted_time)
        
        model.fit(X,y)
        self.model = model
        
        self.sympy_compatible_phenotype = model.sympy_compatible_phenotype

        return model
    
    def predict(self, X):
        assert self.model != None
        y_pred = self.model.predict(X)
        
        return y_pred 
        
    def score(self, X, y):
        self.model.score(X, y)

        
        


est = OptimisedGPRegressor(
        population_size = 250,
        n_novelties = 10,
        favor_less_deep_trees = True,
                   )

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

def pre_train_fn(est, X, y): 
    """set max_time in seconds based on length of X."""
    slack = 20
    if len(X)<=1000:
        max_time = 3600 - slack
    else:
        max_time = 36000 - slack
    est.set_params(max_time=max_time)

# pass the function to eval_kwargs
eval_kwargs = dict(
    pre_train=pre_train_fn,
    test_params={'max_time': 100,
                 }
)

