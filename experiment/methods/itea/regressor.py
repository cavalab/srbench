import os

os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"

import pyITEA as itea

hyper_params = [
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 5),),
        "transfunctions": ("[Id, Sin]",),
        "npop": (1000,),
        "ngens": (1000,),
    },
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Sin]",),
        "npop": (1000,),
        "ngens": (1000,),
        "nonzeroexps": (10,),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Sin]",),
        "npop": (1000,),
        "ngens": (1000,),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 5),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
        "npop": (1000,),
        "ngens": (1000,),
    },
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
        "npop": (1000,),
        "ngens": (1000,),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
        "npop": (1000,),
        "ngens": (1000,),
        "nonzeroexps": (10,),
    },
]


def pre_train(est, X, y):
    """Adjusting ngens to 500 if dataset is too large"""

    # adjust generations based onsize of X versus batch size
    if len(X) >= 10_000:
        est.ngens = 250
    print("ITEA ngens adjusted to", est.gens)


# Create the pipeline for the model
eval_kwargs = {"scale_x": False, "scale_y": False}
est = itea.ITEARegressor(
    npop=1000,
    ngens=500,
    exponents=(-1, 1),
    termlimit=(2, 2),
    nonzeroexps=1,
    transfunctions="[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",
)


def complexity(e):
    return e.len


def model(e, X):
    # get sympy compatible
    new_model = e.sympy.replace("^", "**")

    new_model = new_model.replace("sqrtAbs", "PSQRT")

    for i, f in reversed(list(enumerate(X.columns))):
        new_model = new_model.replace(f"x{i}", f)

    return new_model
