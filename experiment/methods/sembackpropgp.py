from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'popsize': (250,), 'generations': (2000,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (2000,), 'generations': (250,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (1000,), 'generations': (500,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (250,), 'generations': (2000,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
    {
        'popsize': (2000,), 'generations': (250,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
    {
        'popsize': (250,), 'generations': (2000,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
]

est = GPG( popsize=1000, generations=100, time=8*60*60, evaluations=-1, 
    linearscaling=True, functions='+_-_*_aq', erc=True, initmaxtreeheight=6, 
    maxtreeheight=15, maxsize=1000, 
    subcross=0.0, sbagx=False,
    sbrdo=0.75, submut=0.25,
    sblibtype='p_15_9999_l_n',
    unifdepthvar=True, 
    tournament=4,
    caching=False,
    gomea=False, ims=False, silent=True, parallel=False )

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
