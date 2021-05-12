from ..EPLEXRegressor import complexity,model
from ellyn import ellyn
from .params._eplexregressor import params


# Create the pipeline for the model
est = ellyn(selection='epsilon_lexicase',
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
            )

est.set_params(**params)
est.op_list = ['n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos']

# double the evals
est.g *= 2**0.5
est.popsize *= 2**0.5
