"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""

import numpy as np

from .operator_eval import forward_eval_function, reverse_eval_function

ENGINE = "Python"


def evaluate(stack, x, constants):
    """Evaluate an equation

    Evaluate the equation associated with an Agraph, at the values x.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation

    Returns
    -------
    Mx1 array of numeric
        :math`f(x)`
    """
    forward_eval = _forward_eval(stack, x, constants)
    return forward_eval[-1].reshape((-1, 1))


def evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c):
    """Evaluate equation and take derivative

    Evaluate the derivatives of the equation associated with an Agraph, at the
    values x.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation
    wrt_param_x_or_c : boolean
        Take derivative with respect to x or constants. True signifies
        derivatives are wrt x. False signifies derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric:
        Derivatives of all dimensions of x/constants at location x.
    """
    return _evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c)


def _forward_eval(stack, x, constants):
    forward_eval = np.empty((stack.shape[0], x.shape[0]))
    for i, (node, param1, param2) in enumerate(stack):
        forward_eval[i] = forward_eval_function(node, param1, param2, x,
                                                constants, forward_eval)
    return forward_eval


def _evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c):

    forward_eval = _forward_eval(stack, x, constants)

    if wrt_param_x_or_c:  # x
        deriv_shape = x.shape
        deriv_wrt_node = 0
    else:  # c
        deriv_shape = (x.shape[0], len(constants))
        deriv_wrt_node = 1

    derivative = _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval,
                               stack)

    return forward_eval[-1].reshape((-1, 1)), derivative


def _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval, stack):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        node, param1, param2 = stack[i]
        if node == deriv_wrt_node:
            derivative[:, param1] += reverse_eval[i]
        else:
            reverse_eval_function(node, i, param1, param2, forward_eval,
                                  reverse_eval)
    return derivative
