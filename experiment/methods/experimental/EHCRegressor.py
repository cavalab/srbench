from ellyn import ellyn

# 500,000 evaluations = 100,000 with 1 constant hill climbing iteration and 3
# EHC 
pop_sizes = [100, 500, 1000]
gs = [1000, 200, 100]
op_lists=[
        ['n','v','+','-','*','/','exp','log','2','3', 'sqrt'],
        ['n','v','+','-','*','/', 'exp','log','2','3', 'sqrt',
         'sin','cos']
        ]

hyper_params = []

for p, g in zip(pop_sizes, gs):
    for op_list in op_lists:
        hyper_params.append({
                'popsize':[p],
                'g':[g],
                'op_list':[op_list]
                })


# Create the pipeline for the model
est = ellyn(
            eHC_on=True,
            eHC_its=3,
            selection='afp',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=False,
            num_islands=10,
            island_gens=100,
            verbosity=1,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True,
            max_len = 64,
            max_len_init=20,
            popsize=1000,
            g=250
            )

def complexity(est):
    return len(est.best_estimator_)

def model(est):
    return est.stack_2_eqn(est.best_estimator_)
