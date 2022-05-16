import numpy as np

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend.expression import Expression
from bingo.symbolic_regression.agraph.simplification_backend.interpreter import build_cas_expression, build_agraph_stack


def test_interpreter_round_trip():
    stack = np.array([[CONSTANT, -1, -1],
                      [CONSTANT, -1, -1],
                      [SUBTRACTION, 0, 1],
                      [VARIABLE, 0, 0],
                      [ADDITION, 3, 3],
                      [SUBTRACTION, 4, 4],
                      [MULTIPLICATION, 2, 5],
                      [SUBTRACTION, 6, 2],
                      ])
    cas_expression = build_cas_expression(stack)

    buffer_0 = Expression(CONSTANT, [0])
    buffer_1 = Expression(CONSTANT, [1])
    buffer_2 = Expression(SUBTRACTION, [buffer_0.copy(), buffer_1.copy()])
    buffer_3 = Expression(VARIABLE, [0])
    buffer_4 = Expression(ADDITION, [buffer_3.copy(), buffer_3.copy()])
    buffer_5 = Expression(SUBTRACTION, [buffer_4.copy(), buffer_4.copy()])
    buffer_6 = Expression(MULTIPLICATION, [buffer_2.copy(), buffer_5.copy()])
    buffer_7 = Expression(SUBTRACTION, [buffer_6.copy(), buffer_2.copy()])
    expected_cas_expression = buffer_7

    assert cas_expression == expected_cas_expression

    interpreted_stack = build_agraph_stack(cas_expression)
    np.testing.assert_array_equal(interpreted_stack, stack)


def test_building_stack_from_large_associative_operators():
    assoc_expression = Expression(MULTIPLICATION,
                                  [Expression(VARIABLE, [0]),
                                   Expression(VARIABLE, [1]),
                                   Expression(VARIABLE, [2]),
                                   Expression(VARIABLE, [3]),
                                   Expression(VARIABLE, [4]),
                                   ])
    expected_stack = np.array([[0, 0, 0],
                               [0, 1, 1],
                               [0, 2, 2],
                               [0, 3, 3],
                               [0, 4, 4],
                               [4, 0, 1],
                               [4, 3, 4],
                               [4, 2, 6],
                               [4, 5, 7]])
    interpreted_stack = build_agraph_stack(assoc_expression)

    np.testing.assert_array_equal(interpreted_stack, expected_stack)


def test_building_stack_from_large_associative_operators_w_consts():
    assoc_expression = Expression(MULTIPLICATION,
                                  [Expression(CONSTANT, [0]),
                                   Expression(VARIABLE, [0]),
                                   Expression(VARIABLE, [1]),
                                   Expression(VARIABLE, [2]),
                                   Expression(VARIABLE, [3]),
                                   ])
    expected_stack = np.array([[1, -1, -1],
                               [0, 0, 0],
                               [0, 1, 1],
                               [0, 2, 2],
                               [0, 3, 3],
                               [4, 1, 2],
                               [4, 3, 4],
                               [4, 5, 6],
                               [4, 0, 7]])
    interpreted_stack = build_agraph_stack(assoc_expression)

    np.testing.assert_array_equal(interpreted_stack, expected_stack)


