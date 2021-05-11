from pyGPGOMEA import GPGOMEARegressor as GPG
from .params._gpgomearegressor import params

est = GPG(gomea=True, time=2*60*60, generations=-1, evaluations=500000, 
          ims=False, erc=True, linearscaling=True, silent=True, parallel=False)

est.set_params(**params)

est.functions = '+_-_*_p/_plog_sqrt_sin_cos'

# double the evals
est.evaluations *= 2

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
