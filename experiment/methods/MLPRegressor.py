import xgboost
from sklearn.neural_network import MLPRegressor
hyper_params = [
    {
        'activation' : ('logistic', 'tanh', 'relu',),
        'solver' : ('lbfgs','adam','sgd',),
        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
    },
]

est=MLPRegressor()
