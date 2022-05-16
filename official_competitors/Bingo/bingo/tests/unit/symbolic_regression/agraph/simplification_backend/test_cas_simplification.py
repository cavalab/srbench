# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend.expression import Expression
from bingo.symbolic_regression.agraph.simplification_backend.automatic_simplification import automatic_simplify


@pytest.fixture
def x_var():
    return Expression(VARIABLE, [0, ])


@pytest.fixture
def y_var():
    return Expression(VARIABLE, [1, ])


@pytest.fixture
def negative_one():
    return Expression(INTEGER, [-1, ])


@pytest.fixture
def zero():
    return Expression(INTEGER, [0, ])


@pytest.fixture
def one():
    return Expression(INTEGER, [1, ])


@pytest.fixture
def two():
    return Expression(INTEGER, [2, ])


@pytest.fixture
def x_squared(x_var, two):
    return Expression(POWER, [x_var, two])


@pytest.fixture
def x_inv(x_var, negative_one):
    return Expression(POWER, [x_var, negative_one])


@pytest.fixture
def two_y(y_var, two):
    return Expression(MULTIPLICATION, [two, y_var])


@pytest.fixture
def x_times_y(x_var, y_var):
    return Expression(MULTIPLICATION, [x_var, y_var])


def test_expression_base(x_var, x_squared):
    assert x_var.base == x_var
    assert x_squared.base == x_var


def test_expression_exponent(x_var, x_squared, one, two):
    assert x_var.exponent == one
    assert x_squared.exponent == two


def test_expression_term(x_var, two_y, x_times_y, y_var):
    assert x_var.term == Expression(MULTIPLICATION, [x_var])
    assert two_y.term == Expression(MULTIPLICATION, [y_var])
    assert x_times_y.term == x_times_y


def test_expression_const(x_var, two_y, x_times_y, one, two):
    assert x_var.coefficient == one
    assert two_y.coefficient == two
    assert x_times_y.coefficient == one


def test_expression_order(one, two, x_var, y_var, x_squared, two_y, x_times_y):
    assert one < two
    assert two < x_var
    assert x_var < x_squared
    assert x_squared < y_var
    assert y_var < two_y
    assert two_y < x_times_y


def test_no_simplification(one, two, x_var, y_var):
    assert automatic_simplify(one) == one
    assert automatic_simplify(two) == two
    assert automatic_simplify(x_var) == x_var
    assert automatic_simplify(y_var) == y_var


def test_power_simplification(zero, one, two, x_var, x_squared, x_times_y,
                              y_var):
    x_to_zero = Expression(POWER, [zero, x_var])
    assert automatic_simplify(x_to_zero) == x_to_zero
    assert automatic_simplify(Expression(POWER, [one, x_var])) == one
    assert automatic_simplify(Expression(POWER, [x_var, one])) == x_var
    assert automatic_simplify(Expression(POWER, [x_var, zero])) == one
    four = Expression(INTEGER, [4])
    assert automatic_simplify(Expression(POWER, [two, two])) == four

    x_squared_squared = Expression(POWER, [x_squared, two])
    x_4 = Expression(POWER, [x_var, four])
    assert automatic_simplify(x_squared_squared) == x_4

    xy_squared = Expression(POWER, [x_times_y, two])
    y_squared = Expression(POWER, [y_var, two])
    x_squared_y_squared = Expression(MULTIPLICATION, [x_squared, y_squared])
    assert automatic_simplify(xy_squared) == x_squared_y_squared


def test_product_simplification(zero, one, two, x_var, x_squared, x_times_y,
                                y_var, x_inv):
    one_two_zero = Expression(MULTIPLICATION, [one, two, zero])
    assert automatic_simplify(one_two_zero) == zero
    x_one_zero = Expression(MULTIPLICATION, [x_var, two, zero])
    assert automatic_simplify(x_one_zero) == zero
    unary_mult = Expression(MULTIPLICATION, [x_var])
    assert automatic_simplify(unary_mult) == x_var

    x_x_inv = Expression(MULTIPLICATION, [x_var, x_inv])
    assert automatic_simplify(x_x_inv) == one
    x_y_x_inv = Expression(MULTIPLICATION, [x_var, y_var, x_inv])
    assert automatic_simplify(x_y_x_inv) == y_var
    x_y_x = Expression(MULTIPLICATION, [x_var, y_var, x_var])
    x_squared_y = Expression(MULTIPLICATION, [x_squared, y_var])
    assert automatic_simplify(x_y_x) == x_squared_y
    x_times_y_2 = Expression(MULTIPLICATION,
                             [x_var,
                              Expression(MULTIPLICATION, [y_var, two])])
    x_y_2 = Expression(MULTIPLICATION, [two, x_var, y_var])
    assert automatic_simplify(x_times_y_2) == x_y_2


def test_sum_simplification(zero, one, two, x_var, y_var, negative_one):
    one_p_one = Expression(ADDITION, [one, one])
    assert automatic_simplify(one_p_one) == two
    x_p_zero = Expression(ADDITION, [x_var, zero])
    assert automatic_simplify(x_p_zero) == x_var
    unary_add = Expression(ADDITION, [x_var])
    assert automatic_simplify(unary_add) == x_var

    neg_x = Expression(MULTIPLICATION, [negative_one, x_var])
    x_p_neg_x = Expression(ADDITION, [x_var, neg_x])
    assert automatic_simplify(x_p_neg_x) == zero
    x_p_y_p_neg_x = Expression(ADDITION, [x_var, y_var, neg_x])
    assert automatic_simplify(x_p_y_p_neg_x) == y_var
    x_p_y_p_x = Expression(ADDITION, [x_var, y_var, x_var])
    two_x = Expression(MULTIPLICATION, [two, x_var])
    two_x_p_y = Expression(ADDITION, [two_x, y_var])
    assert automatic_simplify(x_p_y_p_x) == two_x_p_y
    x_p_y_p_2 = Expression(ADDITION,
                           [x_var,
                            Expression(ADDITION, [y_var, two])])
    two_p_x_p_y = Expression(ADDITION, [two, x_var, y_var])
    assert automatic_simplify(x_p_y_p_2) == two_p_x_p_y


def test_quotient_simplification(zero, one, two, x_var, x_squared, x_inv,
                                 negative_one):
    one_half = Expression(DIVISION, [one, two])
    two_inv = Expression(POWER, [two, negative_one])
    assert automatic_simplify(one_half) == two_inv

    x_over_0 = Expression(DIVISION, [x_var, zero])
    zero_inv = Expression(POWER, [zero, negative_one])
    zero_inv_x = Expression(MULTIPLICATION, [zero_inv, x_var])
    assert automatic_simplify(x_over_0) == zero_inv_x

    x_squared_over_x = Expression(DIVISION, [x_squared, x_var])
    assert automatic_simplify(x_squared_over_x) == x_var
    x_over_x_squared = Expression(DIVISION, [x_var, x_squared])
    assert automatic_simplify(x_over_x_squared) == x_inv


def test_difference_simplification(zero, one, two, x_var, x_squared, x_inv,
                                   negative_one):
    one_minus_2 = Expression(SUBTRACTION, [one, two])
    assert automatic_simplify(one_minus_2) == negative_one

    x_minus_0 = Expression(SUBTRACTION, [x_var, zero])
    assert automatic_simplify(x_minus_0) == x_var

    two_x = Expression(MULTIPLICATION, [two, x_var])
    two_x_minus_x = Expression(SUBTRACTION, [two_x, x_var])
    assert automatic_simplify(two_x_minus_x) == x_var

    x_p_x = Expression(ADDITION, [x_var, x_var])
    x_p_x_minus_x = Expression(SUBTRACTION, [x_p_x, x_var])
    assert automatic_simplify(x_p_x_minus_x) == x_var

    x_minus_two_x = Expression(SUBTRACTION, [x_var, two_x])
    negative_x = Expression(MULTIPLICATION, [negative_one, x_var])
    assert automatic_simplify(x_minus_two_x) == negative_x


def test_trig_simplification(zero, one):
    sin_zero = Expression(SIN, [zero, ])
    assert automatic_simplify(sin_zero) == zero

    cos_zero = Expression(COS, [zero, ])
    assert automatic_simplify(cos_zero) == one


def test_log_simplification(zero, one, two):
    log_one = Expression(LOGARITHM, [one, ])
    assert automatic_simplify(log_one) == zero

    exp_2 = Expression(EXPONENTIAL, [two, ])
    log_exp_2 = Expression(LOGARITHM, [exp_2, ])
    assert automatic_simplify(log_exp_2) == two


def test_exp_simplification(zero, one):
    exp_zero = Expression(EXPONENTIAL, [zero, ])
    assert automatic_simplify(exp_zero) == one
