# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.string_generation \
    import get_formatted_string


@pytest.fixture
def all_funcs_command_array():
    return np.array([[INTEGER, 5, 5],
                     [VARIABLE, 0, 0],
                     [CONSTANT, 0, 0],
                     [ADDITION, 1, 0],
                     [SUBTRACTION, 2, 3],
                     [MULTIPLICATION, 4, 1],
                     [DIVISION, 5, 1],
                     [SIN, 6, 0],
                     [COS, 7, 0],
                     [EXPONENTIAL, 8, 0],
                     [LOGARITHM, 9, 0],
                     [POWER, 10, 0],
                     [ABS, 11, 0],
                     [SQRT, 12, 0]])


def test_latex_format(all_funcs_command_array):
    expected_string = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (2.0 - (X_0 + " +\
                      "5))(X_0) }{ X_0 } } } } })^{ (5) }| }"
    generated_string = get_formatted_string("latex", all_funcs_command_array,
                                            constants=[2.0, ])
    assert generated_string == expected_string


def test_console_format(all_funcs_command_array):
    expected_string = "sqrt(|(log(exp(cos(sin(((2.0 - (X_0 + 5))(X_0))/" +\
                      "(X_0) )))))^(5)|)"
    generated_string = get_formatted_string("console", all_funcs_command_array,
                                            constants=[2.0, ])
    assert generated_string == expected_string


def test_stack_format(all_funcs_command_array):
    expected_string = "(0) <= 5 (integer)\n" + \
                      "(1) <= X_0\n" + \
                      "(2) <= C_0 = 2.0\n" + \
                      "(3) <= (1) + (0)\n" + \
                      "(4) <= (2) - (3)\n" + \
                      "(5) <= (4) * (1)\n" + \
                      "(6) <= (5) / (1) \n" + \
                      "(7) <= sin (6)\n" + \
                      "(8) <= cos (7)\n" + \
                      "(9) <= exp (8)\n" + \
                      "(10) <= log (9)\n" + \
                      "(11) <= (10) ^ (0)\n" + \
                      "(12) <= abs (11)\n" + \
                      "(13) <= sqrt (12)\n"
    generated_string = get_formatted_string("stack", all_funcs_command_array,
                                            constants=[2.0, ])
    assert generated_string == expected_string


def test_latex_format_no_consts(all_funcs_command_array):
    expected_string = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (? - (X_0 + " +\
                      "5))(X_0) }{ X_0 } } } } })^{ (5) }| }"
    generated_string = get_formatted_string("latex", all_funcs_command_array,
                                            constants=[])
    assert generated_string == expected_string


def test_console_format_no_consts(all_funcs_command_array):
    expected_string = "sqrt(|(log(exp(cos(sin(((? - (X_0 + 5))(X_0))/" +\
                      "(X_0) )))))^(5)|)"
    generated_string = get_formatted_string("console", all_funcs_command_array,
                                            constants=[])
    assert generated_string == expected_string


def test_stack_format_no_consts(all_funcs_command_array):
    expected_string = "(0) <= 5 (integer)\n" + \
                      "(1) <= X_0\n" + \
                      "(2) <= C\n" + \
                      "(3) <= (1) + (0)\n" + \
                      "(4) <= (2) - (3)\n" + \
                      "(5) <= (4) * (1)\n" + \
                      "(6) <= (5) / (1) \n" + \
                      "(7) <= sin (6)\n" + \
                      "(8) <= cos (7)\n" + \
                      "(9) <= exp (8)\n" + \
                      "(10) <= log (9)\n" + \
                      "(11) <= (10) ^ (0)\n" + \
                      "(12) <= abs (11)\n" + \
                      "(13) <= sqrt (12)\n"
    generated_string = get_formatted_string("stack", all_funcs_command_array,
                                            constants=[])
    assert generated_string == expected_string

