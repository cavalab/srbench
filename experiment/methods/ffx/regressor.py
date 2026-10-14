from __future__ import annotations

from typing import Any

from ffx import FFXRegressor

# FFX has no parameters!
hyper_params: list[dict[str, Any]] = [
    {},
]

est = FFXRegressor()


def complexity(est):
    return est.model_.complexity()


def model(est, X=None):
    return est.model_.str2().replace("^", "**")
