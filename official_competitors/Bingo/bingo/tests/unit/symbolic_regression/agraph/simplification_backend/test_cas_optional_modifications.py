from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend.expression import Expression
from bingo.symbolic_regression.agraph.simplification_backend.optional_expression_modification import optional_modifications


def test_inserting_subtraction():
    negative_one = Expression(INTEGER, [-1])
    x = [Expression(VARIABLE, [i, ]) for i in range(4)]
    neg_x = [Expression(MULTIPLICATION, [negative_one, xx]) for xx in x]
    x01_plus_negx23 = Expression(ADDITION, x[:2] + neg_x[2:])
    x01_minus_x23 = Expression(SUBTRACTION,
                               [Expression(ADDITION, x[:2]),
                                Expression(ADDITION, x[2:])])
    assert optional_modifications(x01_plus_negx23) == x01_minus_x23


def test_inserting_subtraction_no_addition():
    negative_one = Expression(INTEGER, [-1])
    x = [Expression(VARIABLE, [i, ]) for i in range(2)]
    neg_x = [Expression(MULTIPLICATION, [negative_one, xx]) for xx in x]
    nx0_plus_nx1 = Expression(ADDITION, neg_x)
    neg_x01 = Expression(MULTIPLICATION,
                         [negative_one,
                          Expression(ADDITION, x)])
    assert optional_modifications(nx0_plus_nx1) == neg_x01


def test_round_trip_with_integer_power():
    x_to_5 = Expression(POWER,
                        [Expression(VARIABLE, [0]),
                         Expression(INTEGER, [5])])
    xxxxx = Expression(MULTIPLICATION, [Expression(VARIABLE, [0])] * 5)
    assert optional_modifications(x_to_5) == xxxxx
