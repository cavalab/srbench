from sklearn import svm

hyper_params = [
    {
        'C': (1e-06,1e-04,0.1,1,),
        'loss' : ('epsilon_insensitive','squared_epsilon_insensitive',),
    },
]

est=svm.LinearSVR()

complexity = None
