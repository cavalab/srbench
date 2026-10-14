from ellyn import ellyn

# 500,000 evaluations = 100,000 with 1 eval, 1 constant hill climbing, and 3
# EHC iterations
pop_sizes = [250, 500, 1000]
gs = [2500, 1000, 500]
op_lists = [
    ["n", "v", "+", "-", "*", "/", "exp", "log", "2", "3", "sqrt"],
    ["n", "v", "+", "-", "*", "/", "exp", "log", "2", "3", "sqrt", "sin", "cos"],
]

hyper_params = []

# We should tune budget-related stuff (e.g. popsize and generations) only if
# there is a time limit setting
for p, g in zip(pop_sizes, gs):
    for op_list in op_lists:
        hyper_params.append({"popsize": [p], "g": [g], "op_list": [op_list]})


# Create the pipeline for the model
est = ellyn(
    eHC_on=True,
    eHC_its=3,
    selection="afp",
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
    max_len=64,
    max_len_init=20,
    popsize=1000,
    g=100,
    time_limit=60 * 60,
)


def complexity(est):
    return len(est.best_estimator_)


def model(est, X=None):
    model_str = est.stack_2_eqn(est.best_estimator_)

    # protected sqrt uses |cdot|. removing it
    model_str = model_str.replace("|", "")

    return model_str


eval_kwargs = dict(use_dataframe=False)
