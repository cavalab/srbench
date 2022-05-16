"""
This module contains encapsulates the ability to translate equations in the
form of AGraphs (specified by a command array) to/from computer algebra system
`Expressions` which are used in the simplification process.
"""

import numpy as np

from ..operator_definitions \
    import IS_TERMINAL_MAP, IS_ARITY_2_MAP, CONSTANT, INTEGER, VARIABLE
from .expression import Expression


def build_cas_expression(agraph_stack):
    """Translate a command array into computer algebra system (CAS) expression

    Parameters
    ----------
    agraph_stack: AGraph command array
        The command array that encodes an equation

    Returns
    -------
        CAS Expression

    """
    return _build_expresion_recursive(agraph_stack, len(agraph_stack) - 1)


def _build_expresion_recursive(stack, location):
    operator, param_1, param_2 = stack[location]
    if IS_TERMINAL_MAP[operator]:
        if operator == CONSTANT:
            param_1 = param_2 = location
        operands = [param_1]
        if IS_ARITY_2_MAP[operator]:
            operands += [param_2]
        return Expression(operator, operands)

    operands = [_build_expresion_recursive(stack, param_1)]
    if IS_ARITY_2_MAP[operator]:
        operands += [_build_expresion_recursive(stack, param_2)]
    return Expression(operator, operands)


def build_agraph_stack(expression):
    """Translate computer algebra system (CAS) expression into a command array

    Parameters
    ----------
    agraph_stack: `Expression`
        A computer algebra system expression

    Returns
    -------
        A command array

    """
    stack_dict = {}
    _build_stack_recursive(expression, stack_dict)
    stack = np.empty((len(stack_dict), 3), dtype=int)
    for command, loc in stack_dict.items():
        if command[0] == CONSTANT:
            stack[loc] = (CONSTANT, -1, -1)
        else:
            stack[loc] = command
    return stack


def _build_stack_recursive(expression, stack_dict):
    if expression.operator in [INTEGER, CONSTANT, VARIABLE]:
        command = (expression.operator, expression.operands[0],
                   expression.operands[0])
        return _add_command_to_stack_dict(command, stack_dict)

    operand_locations = [_build_stack_recursive(operand, stack_dict)
                         for operand in expression.operands]

    if len(operand_locations) == 1:
        command = (expression.operator, operand_locations[0],
                   operand_locations[0])
        return _add_command_to_stack_dict(command, stack_dict)

    if len(operand_locations) == 2:
        command = (expression.operator, operand_locations[0],
                   operand_locations[1])
        return _add_command_to_stack_dict(command, stack_dict)

    if not expression.is_constant_valued and \
            expression.operands[0].is_constant_valued:
        loc = _add_associative_operators_to_stack(expression.operator,
                                                  operand_locations[1:],
                                                  stack_dict)
        command = (expression.operator, operand_locations[0],
                   loc)
        return _add_command_to_stack_dict(command, stack_dict)

    return _add_associative_operators_to_stack(expression.operator,
                                               operand_locations, stack_dict)


def _add_command_to_stack_dict(command, stack_dict):
    if command in stack_dict:
        return stack_dict[command]
    loc = len(stack_dict)
    stack_dict[command] = loc
    return loc


def _add_associative_operators_to_stack(operator, operand_locs, stack_dict):
    if len(operand_locs) == 1:
        return operand_locs[0]
    operand_div = len(operand_locs) // 2
    loc_1 = _add_associative_operators_to_stack(operator,
                                                operand_locs[:operand_div],
                                                stack_dict)
    loc_2 = _add_associative_operators_to_stack(operator,
                                                operand_locs[operand_div:],
                                                stack_dict)
    command = (operator, loc_1, loc_2)
    return _add_command_to_stack_dict(command, stack_dict)
