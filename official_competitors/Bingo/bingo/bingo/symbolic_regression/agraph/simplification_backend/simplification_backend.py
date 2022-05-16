"""
This module represents the python backend associated with `AGraph` equation
simplification.  This backend is used to perform the the simplification of an
equation represented by an `AGraph`.  It can be performed based on a simple
reduction or a more involved algebraic simplification.
"""

import numpy as np

from ..operator_definitions import IS_ARITY_2_MAP, IS_TERMINAL_MAP
from .simplify import simplify as cas_simplify

ENGINE = "Python"


def get_utilized_commands(stack):
    """Find which commands are utilized

    Find the commands (rows) of the stack upon which the last command of the
    stack depends. This is inclusive of the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    list of bool of length N
        Boolean values for whether each command is utilized.
    """
    util = [False]*stack.shape[0]
    util[-1] = True
    for i in range(1, stack.shape[0]):
        node, param1, param2 = stack[-i]
        if util[-i] and not IS_TERMINAL_MAP[node]:
            util[param1] = True
            if IS_ARITY_2_MAP[node]:
                util[param2] = True
    return util


def simplify_stack(stack):
    """Simplifies a stack based on computational algebra

    An acyclic graph is given in stack form.  The stack is algebraically
    simplified and put in a canonical form.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    Mx3 numpy array of int. :
        a simplified stack representing the original equation
    """
    return cas_simplify(stack)


def reduce_stack(stack):
    """Reduces a stack

    An acyclic graph is given in stack form.  The stack is simplified to
    consist only of the commands used by the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    Mx3 numpy array of int. :
        a simplified stack where M is the number of  used commands
    """
    used_commands = get_utilized_commands(stack)
    reduced_param_map = {}
    num_commands = np.sum(used_commands)
    new_stack = np.empty((num_commands, 3), int)
    j = 0
    for i, (node, param1, param2) in enumerate(stack):
        if used_commands[i]:
            new_stack[j, 0] = node
            if IS_TERMINAL_MAP[node]:
                new_stack[j, 1] = param1
                new_stack[j, 2] = param2
            else:
                new_stack[j, 1] = reduced_param_map[param1]
                if IS_ARITY_2_MAP[node]:
                    new_stack[j, 2] = reduced_param_map[param2]
                else:
                    new_stack[j, 2] = new_stack[j, 1]
            reduced_param_map[i] = j
            j += 1
    return new_stack
