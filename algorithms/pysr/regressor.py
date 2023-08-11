from pysr import PySRRegressor
from multiprocessing import cpu_count


def complexity(est):
    return est.get_best()["complexity"]


def model(est):
    return str(est.sympy())


est = PySRRegressor(
    niterations=1_000_000_000,
    ncyclesperiteration=2_500,
    population_size=100,
    populations=max(15, cpu_count()*2),
    # budget 10 minutes for compile time,
    # ensuring we can finish within 2 hours:
    timeout_in_seconds=2*60*60 - 10*60,
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
    procs=cpu_count(),
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
