from pysr import PySRRegressor
from multiprocessing import cpu_count

hyper_params = [
    {
        'niterations' : [1_000_000_000],
        'maxsize' : [20],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [30],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [50],
        'maxdepth' : [20],
    },
    {
        'niterations' : [1_000_000],
        'maxsize' : [30],
        'maxdepth' : [20],
        'population_size' : [200],
    },
]

def complexity(est):
    return int(est.get_best()["complexity"])


def model(est, X=None):
    return str(est.sympy())


est = PySRRegressor(
    niterations=1_000_000_000,
    ncycles_per_iteration=2_500,
    population_size=100,
    populations=2, # originally max(15, 2*n_cpu), but srbench runs on a single core
    verbosity=0,
    # budget 10 minutes for compile time,
    # ensuring we can finish within 2 hours:
    timeout_in_seconds=60*60 - 10*60,
    maxsize=30,
    maxdepth=20,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "exp", "log", "sqrt"],
    constraints={
        **dict(
            sin=9,
            exp=9,
            log=9,
            sqrt=9,
        ),
        **{"/": (-1, 9)}
    },
    nested_constraints=dict(
        sin=dict(
            sin=0,
            exp=1,
            log=1,
            sqrt=1,
        ),
        exp=dict(
            exp=0,
            log=0,
        ),
        log=dict(
            exp=0,
            log=0,
        ),
        sqrt=dict(
            sqrt=0,
        )
    ),
    # prefer multiprocessing:
    procs=1, # cpu_count(),
    multithreading=False,
    batching=True,
    batch_size=50,
    turbo=True,
    weight_optimize=0.001,
    adaptive_parsimony_scaling=1_000.0,
    parsimony=0.0,
)

# See https://astroautomata.com/PySR/tuning/ for tuning advice
hyper_params = [{}]

eval_kwargs = {
    "test_params": dict(
        niterations=3,
        ncycles_per_iteration=500,
        populations=3,
    )
}
