import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend.expression import Expression
from bingo.symbolic_regression.agraph.simplification_backend.constant_folding import fold_constants


@pytest.fixture
def c0():
    return Expression(CONSTANT, [0])


@pytest.fixture
def c1():
    return Expression(CONSTANT, [1])


@pytest.fixture
def one():
    return Expression(INTEGER, [1])


@pytest.fixture()
def x():
    return Expression(VARIABLE, [0])


def test_no_folding_single_constant(c0, x):
    c0_p_x = Expression(ADDITION, [c0.copy(), x.copy()])
    assert fold_constants(c0_p_x) == c0_p_x


def test_no_folding_two_constants(c0, c1, x):
    c0_p_x = Expression(ADDITION, [c0.copy(), x.copy()])
    mult = Expression(MULTIPLICATION, [c1.copy(), c0_p_x.copy()])
    assert fold_constants(mult) == mult


def test_folding_two_constants_in_assocative_operation(c0, c1, x):
    c0_p_c1_p_x = Expression(ADDITION, [c0.copy(), c1.copy(), x.copy()])
    c0_p_x = Expression(ADDITION, [c0.copy(), x.copy()])
    assert fold_constants(c0_p_c1_p_x) == c0_p_x


def test_folding_integer_into_constant(c0, one):
    c0_p_one = Expression(ADDITION, [c0.copy(), one.copy()])
    assert fold_constants(c0_p_one) == c0


def test_folding_integer_into_constant_and_repetition(c0, one):
    c0_p_one = Expression(ADDITION, [c0.copy(), one.copy()])
    c0_p_one_squared = Expression(MULTIPLICATION, [c0_p_one.copy(),
                                                   c0_p_one.copy()])
    assert fold_constants(c0_p_one_squared) == c0


def test_folding_2_consts_to_1(c0, c1, x):
    c0c1 = Expression(MULTIPLICATION, [c0.copy(), c1.copy()])
    two_const_expr = Expression(ADDITION, [c0c1.copy(), x.copy()])
    expected_folded_expression = Expression(ADDITION, [c0.copy(), x.copy()])
    assert fold_constants(two_const_expr) == expected_folded_expression