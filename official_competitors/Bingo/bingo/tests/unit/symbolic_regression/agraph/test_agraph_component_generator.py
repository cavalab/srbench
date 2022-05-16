# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator


@pytest.fixture
def sample_component_generator():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5)
    generator.add_operator(2)
    generator.add_operator(6)
    return generator


@pytest.mark.parametrize("param_name, param_value, expected_error", [
    ("input_x_dimension", -1, ValueError),
    ("input_x_dimension", "string", TypeError),
    ("num_initial_load_statements", 0, ValueError),
    ("num_initial_load_statements", "string", TypeError),
    ("terminal_probability", -0.1, ValueError),
    ("terminal_probability", "string", TypeError),
    ("terminal_probability", 2, ValueError),
    ("constant_probability", -0.1, ValueError),
    ("constant_probability", "string", TypeError),
    ("constant_probability", 2, ValueError)
])
def test_raises_error_invalid_init(param_name, param_value, expected_error):
    kwargs = {"input_x_dimension": 1}
    kwargs[param_name] = param_value
    with pytest.raises(expected_error):
        _ = ComponentGenerator(**kwargs)


def test_raises_error_random_operator_with_no_operators():
    no_operator_generator = ComponentGenerator(input_x_dimension=1,
                                               terminal_probability=0.0)
    _ = no_operator_generator.random_command(0)
    with pytest.raises(IndexError):
        _ = no_operator_generator.random_command(1)
    with pytest.raises(IndexError):
        _ = no_operator_generator.random_operator()


def test_random_terminal():
    np.random.seed(0)
    generator = ComponentGenerator(input_x_dimension=3,
                                   constant_probability=0.25)
    terminals = [generator.random_terminal() for _ in range(100)]
    assert terminals.count(VARIABLE) == 72
    assert terminals.count(CONSTANT) == 28


def test_random_terminal_default_probability_about_25():
    np.random.seed(0)
    generator = ComponentGenerator(input_x_dimension=3)
    terminals = [generator.random_terminal() for _ in range(100)]
    assert terminals.count(VARIABLE) == 72
    assert terminals.count(CONSTANT) == 28


def test_random_operator():
    np.random.seed(0)
    generator = ComponentGenerator(input_x_dimension=3)
    generator.add_operator("addition")
    generator.add_operator("multiplication")
    operators = [generator.random_operator() for _ in range(100)]
    assert operators.count(ADDITION) == 51
    assert operators.count(MULTIPLICATION) == 49


def test_random_operator_parameter():
    generator = ComponentGenerator(input_x_dimension=3)
    for command_location in np.random.randint(1, 100, 50):
        command_param = generator.random_operator_parameter(command_location)
        assert command_param < command_location


def test_random_terminal_parameter():
    generator = ComponentGenerator(input_x_dimension=3)
    for _ in range(20):
        assert generator.random_terminal_parameter(VARIABLE) in [0, 1, 2]
        assert generator.random_terminal_parameter(CONSTANT) == -1


@pytest.mark.parametrize("operator_to_add", [SUBTRACTION, "subtraction", "-"])
def test_add_operator(sample_component_generator, operator_to_add):
    generator = ComponentGenerator(input_x_dimension=3)
    generator.add_operator(operator_to_add)
    assert generator.random_operator() == SUBTRACTION


def test_raises_error_on_invalid_add_operator(sample_component_generator):
    generator = ComponentGenerator(input_x_dimension=3)
    np.random.seed(0)
    with pytest.raises(ValueError):
        generator.add_operator("invalid operator")


def test_add_operator_with_weight():
    generator = ComponentGenerator(input_x_dimension=3)
    generator.add_operator(ADDITION)
    generator.add_operator(SUBTRACTION, operator_weight=0.0)
    operators = [generator.random_operator() for _ in range(100)]
    assert SUBTRACTION not in operators


def test_random_command():
    generator = ComponentGenerator(input_x_dimension=2,
                                   terminal_probability=0.6)
    generator.add_operator(ADDITION)
    generator.add_operator(SUBTRACTION)
    np.random.seed(5)
    generated_commands = np.empty((6, 3))
    expected_commands = np.array([[CONSTANT, -1, -1],
                                  [ADDITION, 0, 0],
                                  [ADDITION, 1, 0],
                                  [SUBTRACTION, 0, 1],
                                  [VARIABLE, 0, 0],
                                  [VARIABLE, 0, 0]])
    for stack_location in range(generated_commands.shape[0]):
        generated_commands[stack_location, :] = \
            generator.random_command(stack_location)
    np.testing.assert_array_equal(generated_commands, expected_commands)


def test_numbers_of_terminals_and_params():
    generator = ComponentGenerator(input_x_dimension=3)
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 0
    generator.add_operator(ADDITION)
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 1
    generator.add_operator(SUBTRACTION)
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 2
