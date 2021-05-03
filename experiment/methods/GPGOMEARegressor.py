from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'initmaxtreeheight' : (4,), 'functions':('+_-_*_p/_plog_sqrt_sin_cos',),
        'popsize': (1000,), 'linearscaling':(True,),
    },
    { 
        'initmaxtreeheight' : (6,), 'functions':('+_-_*_p/_plog_sqrt_sin_cos',),
        'popsize': (1000,), 'linearscaling':(True,),
    },
    {
        'initmaxtreeheight' : (4,), 'functions':('+_-_*_p/',),
        'popsize': (1000,), 'linearscaling':(True,),
    }, 
    {
        'initmaxtreeheight' : (6,), 'functions':('+_-_*_p/',),
        'popsize': (1000,), 'linearscaling':(True,),
    },
    {
        'initmaxtreeheight' : (4,), 'functions':('+_-_*_p/_plog_sqrt_sin_cos',),
        'popsize': (1000,), 'linearscaling':(False,),
    },
    {
        'initmaxtreeheight' : (6,), 'functions':('+_-_*_p/_plog_sqrt_sin_cos',),
        'popsize': (1000,), 'linearscaling':(False,),
    },
]

est = GPG(gomea=True, time=-1, generations=-1, evaluations=500000, ims=False,
          erc=True, linearscaling=True,
          silent=True, parallel=False)

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
