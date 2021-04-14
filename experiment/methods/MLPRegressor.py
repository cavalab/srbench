from sklearn.neural_network import MLPRegressor
import numpy as np

hyper_params = [
    {
        'activation' : ('logistic', 'tanh', 'relu',),
        'solver' : ('lbfgs','adam','sgd',),
        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
    },
]

est=MLPRegressor()

def complexity(est):
    return np.sum([c.size for c in est.coefs_]
                  + [c.size for c in est.intercepts_])
model = None
