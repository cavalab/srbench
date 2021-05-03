from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'popsize': (1000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (True,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (4,), 'maxsize': (1000,),
    },
    {
        'popsize': (1000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (True,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (8,), 'maxsize': (1000,),
    },
    {
        'popsize': (1000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (True,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (4,), 'maxsize': (5000,),
    },
    {
        'popsize': (1000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (True,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (8,), 'maxsize': (5000,),
    },
    {
        'popsize': (1000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (False,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (4,), 'maxsize': (250,), 'sblibtype': ('p_6_20000',), 
    },
    {
        'popsize': (10000,), 'functions' : ('+_-_*_aq_plog_sin_cos',), 'linearscaling' : (False,),
        'sbrdo': (0.9,), 'submut': (0.1,), 'tournament': (8,), 'maxsize': (250,), 'sblibtype': ('p_6_20000',),
    },
]

# Note: max size and max tree height are TBD I guess.
est = GPG( popsize=100, generations=-1, time=-1, evaluations=500000, 
    linearscaling=True, functions='+_-_*_aq_plog_sin_cos', erc=True, initmaxtreeheight=6, 
    maxtreeheight=20, maxsize=1000, 
    subcross=0.0, sbagx=False,
    sbrdo=0.75, submut=0.25,
    sblibtype='p_10_9999_l_n',
    unifdepthvar=True, 
    tournament=4,
    caching=False,
    gomea=False, ims=False, silent=True, parallel=False )

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
