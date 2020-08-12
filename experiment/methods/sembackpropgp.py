from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'popsize': (100,), 'generations': (1000,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (1000,), 'generations': (100,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (500,), 'generations': (200,),
        'sbrdo': (1.0,), 'submut': (0.0,),
    },
    {
        'popsize': (100,), 'generations': (1000,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
    {
        'popsize': (1000,), 'generations': (100,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
    {
        'popsize': (500,), 'generations': (200,),
        'sbrdo': (0.75,), 'submut': (0.25,),
    },
]

est = GPG( popsize=1000, generations=100, time=-1, evaluations=-1, 
    linearscaling=True, functions='+_-_*_aq', erc=True, initmaxtreeheight=6, 
    maxtreeheight=15, maxsize=1000, 
    subcross=0.0, sbagx=False,
    sbrdo=0.75, submut=0.25,
    sblibtype='p_15_9999_l_n',
    unifdepthvar=True, 
    tournament=4,
    caching=False,
    gomea=False, ims=False, silent=True, parallel=False )
