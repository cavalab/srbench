from ellyn import ellyn
from .params._fe_afpregressor import params

# Create the pipeline for the model
est = ellyn(selection='afp',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=0,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            EstimateFitness=True,
            FE_pop_size=100,
            FE_ind_size=10,
            FE_train_size=10,
            FE_train_gens=10,
            FE_rank=True,
            )

est.set_params(**params)
est.op_list = ['n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos']
# double the evals
est.g *= 2**0.5
est.popsize *= 2**0.5

def complexity(est):
    return len(est.best_estimator_)

def model(est):
    return est.stack_2_eqn(est.best_estimator_)

def pre_train(est, X, y):
    """Adjust settings based on data before training"""
    # adjust generations based on size of X versus FE size
    g = est.g
    est.g = g*len(X)/est.FE_ind_size
    print('FE ellyn gens adjusted from',g,'to',est.g)

eval_kwargs = dict(pre_train=pre_train)
