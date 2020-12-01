from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'initmaxtreeheight' : (4,)
        'evaluations' : (100000, 1000000, 10000000, ),
    },
    {
        'initmaxtreeheight' : (5,)
        'evaluations' : (100000, 1000000, 10000000, ),
    },
    {
        'initmaxtreeheight' : (6,)
        'evaluations' : (100000, 1000000, 10000000, ),
    },
]

est = GPG( gomea=True, time=-1, generations=-1, evaluations=1000000, ims='5_1', 
          silent=True, parallel=False )

def complexity(est):
    return est.get_n_nodes()
