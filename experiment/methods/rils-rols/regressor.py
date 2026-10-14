import sympy as sp
from rils_rols.rils_rols import RILSROLSRegressor
from sklearn.base import RegressorMixin
from symbolic_utils import complexity as cplx

eval_kwargs = {"scale_x": False, "scale_y": False}

complexity_penalty = [0.001, 0.01, 0.1]
max_complexity = [25, 50, 100]

hyper_params = []
for c in complexity_penalty:
    for max_c in max_complexity:
        hyper_params.append(
            {
                "complexity_penalty": [c],
                "max_complexity": [max_c],
            }
        )

est: RegressorMixin = RILSROLSRegressor(
    max_time=60 * 60,
    max_fit_calls=1000 * 1000,
    max_complexity=50,
    sample_size=0,
    verbose=False,
)


def model(est, X=None) -> str:
    if X is None:
        return str(est.model_string())
    else:
        mapping = {"x" + str(i): k for i, k in enumerate(X.columns)}
        new_model = str(est.model_string())
        for k, v in reversed(mapping.items()):
            new_model = new_model.replace(k, v)
        return new_model


def complexity(est):
    return cplx(sp.parse_expr(str(est.model_string())))


def get_population(est) -> list[RegressorMixin]:
    return [est]


def get_best_solution(est) -> RegressorMixin:
    return est
