from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'initmaxtreeheight' : (4,5,6),
    },
]

est = GPG(gomea=True, time=28800, generations=-1, evaluations=500000, 
          ims='5_1', 
          silent=True, parallel=False )

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
