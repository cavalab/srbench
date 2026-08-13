#!/usr/bin/env python3
"""Validate that each benchmarked method is wired up correctly.

SRBench splits every method across two directories:

    algorithms/<name>/          install spec, baked into the docker image at build time
    experiment/methods/<name>/  regressor.py, bind-mounted over /srbench at run time

Both are required. A method that lands in only one of them can still show a full
set of green checks -- the `build-and-test` matrix is generated from `ls algorithms/`,
so a submission that touches only `experiment/methods/` never gets a job at all.
This script closes that gap.

Run it locally before opening a PR:

    python scripts/check_method_layout.py

Exit status is 0 when every check passes, 1 otherwise. Warnings never fail the run.
"""

from __future__ import annotations

import ast
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - only hit outside the srbench env
    yaml = None

ALG_DIR = "algorithms"
METHOD_DIR = os.path.join("experiment", "methods")

# Directories under experiment/methods/ that intentionally have no algorithms/
# counterpart: 2021-era estimator variants, sklearn baselines, and the tuned
# estimators used by `analyze.py -tuned`. They are not containerized and are not
# part of the current benchmark roster. Do not add new entries here -- a new
# method needs both directories. See open issue #161 on retiring these.
LEGACY_METHOD_DIRS = {
    "afp_ehc",
    "afp_fe",
    "experimental",
    "geneticengine_1p1",
    "geneticengine_hc",
    "geneticengine_rs",
    "sklearn_adaboost",
    "sklearn_lasso",
    "sklearn_linear",
    "sklearn_mlp",
    "sklearn_randomforest",
    "sklearn_ridge",
    "sklearn_sgd",
    "tuned",
}

# metadata.yml predates any validation and 10 of these files are empty. Existing
# methods are grandfathered so CI stays green; new methods must fill it in.
# Shrinking this set is a good standalone cleanup PR.
METADATA_GRANDFATHERED = {
    "bsr",
    "eplex",
    "ffx",
    "gplearn",
    "itea",
    "lightgbm",
    "nesymres",
    "sklearn",
    "tir",
    "xgboost",
}

REQUIRED_METADATA_KEYS = ("name", "authors", "email", "description", "url")

errors: list[str] = []
warnings: list[str] = []


def error(method: str, msg: str) -> None:
    errors.append(f"{method}: {msg}")


def warn(method: str, msg: str) -> None:
    warnings.append(f"{method}: {msg}")


def top_level_names(path: str) -> set[str] | None:
    """Names bound at module scope, without importing (imports would need deps)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as exc:
        error(path, f"regressor.py is not valid Python: {exc}")
        return None

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            # `est: RegressorMixin = FeatRegressor(...)` -- the common style here
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def check_algorithm(name: str) -> None:
    """Every algorithms/<name>/ needs a matching runnable method."""
    alg_path = os.path.join(ALG_DIR, name)

    metadata = os.path.join(alg_path, "metadata.yml")
    if not os.path.isfile(metadata):
        error(name, f"missing {metadata}")
    elif yaml is not None:
        try:
            parsed = yaml.safe_load(open(metadata, encoding="utf-8"))
        except yaml.YAMLError as exc:
            error(name, f"{metadata} is not valid YAML: {exc}")
            parsed = None
        if parsed is not None or name not in METADATA_GRANDFATHERED:
            missing = [k for k in REQUIRED_METADATA_KEYS if k not in (parsed or {})]
            if missing and name not in METADATA_GRANDFATHERED:
                error(name, f"{metadata} is missing required key(s): {', '.join(missing)}")
            elif missing:
                warn(name, f"{metadata} is incomplete (missing {', '.join(missing)})")

    method_path = os.path.join(METHOD_DIR, name)
    regressor = os.path.join(method_path, "regressor.py")
    if not os.path.isdir(method_path):
        error(
            name,
            f"has {alg_path}/ but no {method_path}/ -- regressor.py lives under "
            f"{METHOD_DIR}/<name>/, not in {ALG_DIR}/<name>/",
        )
        return
    if not os.path.isfile(regressor):
        error(name, f"missing {regressor}")
        return

    if not os.path.isfile(os.path.join(method_path, "__init__.py")):
        error(name, f"missing {method_path}/__init__.py (an empty file is fine)")

    names = top_level_names(regressor)
    if names is None:
        return
    for required in ("est", "model"):
        if required not in names:
            error(name, f"{regressor} does not define `{required}` at module level")

    if os.path.isfile(os.path.join(alg_path, "regressor.py")):
        warn(name, f"{alg_path}/regressor.py is ignored; the harness imports {regressor}")
    if os.path.isfile(os.path.join(method_path, "metadata.yml")):
        warn(name, f"{method_path}/metadata.yml is ignored; metadata.yml belongs in {alg_path}/")
    for stray in ("install.sh", "environment.yml", "requirements.txt", "Dockerfile"):
        if os.path.isfile(os.path.join(method_path, stray)):
            error(
                name,
                f"{method_path}/{stray} is never read -- install files belong in {alg_path}/",
            )


def check_orphan_method(name: str) -> None:
    """experiment/methods/<name>/ with no algorithms/<name>/ is never built or tested."""
    error(
        name,
        f"has {METHOD_DIR}/{name}/ but no {ALG_DIR}/{name}/ -- the CI matrix is built "
        f"from `ls {ALG_DIR}/`, so this method is never built or tested",
    )


def main() -> int:
    if not os.path.isdir(ALG_DIR) or not os.path.isdir(METHOD_DIR):
        print(f"error: run this from the repository root (missing {ALG_DIR}/ or {METHOD_DIR}/)")
        return 1

    algorithms = sorted(d for d in os.listdir(ALG_DIR) if os.path.isdir(os.path.join(ALG_DIR, d)))
    methods = sorted(
        d
        for d in os.listdir(METHOD_DIR)
        if os.path.isdir(os.path.join(METHOD_DIR, d)) and not d.startswith((".", "__"))
    )

    for name in algorithms:
        check_algorithm(name)

    for name in methods:
        if name not in LEGACY_METHOD_DIRS and name not in set(algorithms):
            check_orphan_method(name)

    print(f"checked {len(algorithms)} algorithms and {len(methods)} method directories")

    for line in warnings:
        print(f"::warning::{line}" if os.environ.get("GITHUB_ACTIONS") else f"warning: {line}")

    if errors:
        print()
        for line in errors:
            print(f"::error::{line}" if os.environ.get("GITHUB_ACTIONS") else f"error: {line}")
        print(f"\n{len(errors)} problem(s) found. See CONTRIBUTING.md for the expected layout.")
        return 1

    print("method layout OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
