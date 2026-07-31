from pybrush import BrushRegressor


kwargs = {
    "verbosity": 1,
    "pop_size": 250,
    "max_gens": 250,
    "max_depth": 8,
    "max_size": 75,
    "initialization": "uniform",
    "validation_size": 0.33,
    "cx_prob": 1 / 7,

    "weights_init": False,
    "mutation_probs": {
        "point": 1 / 6,
        "insert": 1 / 6,
        "delete": 1 / 6,
        "subtree": 1 / 6,
        "toggle_weight_on": 1 / 6,
        "toggle_weight_off": 1 / 6,
    },

    "sel": "lexicase",
    "algorithm": "nsga2",
    "objectives": ["scorer", "complexity"],
    "scorer": "mse",

    "bandit": "dynamic_thompson",
    "num_islands": 1,
    "shuffle_split": True,

    "functions": [
        # Arithmetic
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",

        # Unary functions
        "Sin",
        "Cos",
        "Tanh",
        "Exp",
        "Log",
        "Sqrt",

        # Optional split operators
        "SplitBest",
        "SplitOn",

        # Terminals
        "Constant",
        "Terminal",
    ],
}

est = BrushRegressor(**kwargs)


func_dict = {
    "Mul": "*",
    "Sub": "-",
    "Add": "+",
    "Div": "/",
    "Pow": "**",
}

func_arity = {
    "Mul": 2,
    "Sub": 2,
    "Div": 2,
    "Add": 2,
    "Pow": 2,

    "Sin": 1,
    "Cos": 1,
    "Tanh": 1,
    "Asin": 1,
    "Acos": 1,
    "Sqrt": 1,
    "Log": 1,
    "Exp": 1,
    "Square": 1,
}


def pretify_expr(string, feature_names):
    tokens = string.replace(" ", "").replace(")", "").replace("(", ",").split(",")

    new_string = ""
    stack = []

    for token in tokens:
        stack.append((token, []))

        while len(stack[-1][1]) == func_arity.get(stack[-1][0], 0):
            primitive, args = stack.pop()
            new_string = primitive

            if primitive in func_dict:
                new_string = "(" + func_dict[primitive].join(args) + ")"

            elif "*" in primitive:
                left, right = primitive.split("*", 1)
                stack.append(("Mul", [left]))
                stack.append((right, []))
                continue

            elif primitive not in feature_names:
                try:
                    float(primitive)
                except ValueError:
                    new_string = primitive.lower() + "(" + args[0] + ")"

            if not stack:
                break

            stack[-1][1].append(new_string)

    return new_string


def model(estimator, X=None):
    """
    Return the raw Brush expression string.
    """
    if isinstance(estimator, BrushRegressor):
        return estimator.best_estimator_.get_model()

    # Compatibility with older SRBench/DEAP-style estimators.
    if hasattr(estimator, "model"):
        return estimator.model()

    raise AttributeError("Estimator does not expose a Brush model.")


def complexity(estimator):
    """
    Return the Brush expression size.
    """
    if isinstance(estimator, BrushRegressor):
        return estimator.best_estimator_.fitness.size

    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_.fitness.size

    if hasattr(estimator, "fitness"):
        return estimator.fitness.size

    raise AttributeError("Estimator does not expose model complexity.")