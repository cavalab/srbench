"""Reference linear-regression wrapper for the generic sklearn image."""

from sklearn.linear_model import LinearRegression


est = LinearRegression()
hyper_params = [{"fit_intercept": (True, False)}]


def model(estimator, X=None):
    terms = []
    if estimator.fit_intercept:
        terms.append(str(float(estimator.intercept_)))
    for coefficient, feature in zip(estimator.coef_, X.columns):
        terms.append(f"({float(coefficient)})*({feature})")
    return "+".join(terms) if terms else "0"


def complexity(estimator):
    return len(estimator.coef_) + int(estimator.fit_intercept)
