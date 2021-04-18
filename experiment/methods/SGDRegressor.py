from sklearn import linear_model


hyper_params = [
    {
        'alpha': (1e-06,1e-04,0.01,1,),
        'penalty': ('l2','l1','elasticnet',),
    },
]

est=linear_model.SGDRegressor()

complexity = None
model = None
