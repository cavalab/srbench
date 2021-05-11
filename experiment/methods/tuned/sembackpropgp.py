from pyGPGOMEA import GPGOMEARegressor as GPG
from .params._sembackpropgp import params

# Note: max size and max tree height are TBD I guess.
est = GPG( popsize=100, generations=-1, time=10*60*60, evaluations=500000, 
    linearscaling=True, functions='+_-_*_aq_plog_sin_cos', erc=True, initmaxtreeheight=6, 
    maxtreeheight=20, maxsize=1000, 
    subcross=0.0, sbagx=False,
    sbrdo=0.75, submut=0.25,
    sblibtype='p_10_9999_l_n',
    unifdepthvar=True, 
    tournament=4,
    caching=False,
    gomea=False, ims=False, silent=True, parallel=False )

est.set_params(**params)
est.functions = '+_-_*_aq_plog_sin_cos'

#double the evals
est.evaluations=1000000

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
