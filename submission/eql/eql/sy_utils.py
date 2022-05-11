from sympy import Symbol, simplify, factor, Float, preorder_traversal, Integer
from sympy.parsing.sympy_parser import parse_expr
import math


def simplicity(expr):
    # l0 was too large, return bad score
    if expr.is_number:
        return -10.0
    c = complexity(expr)
    if c <= 5:
        c = 5
    return round(-0.3 * math.log(c / 120, 5), 1)


def complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c


def round_floats(ex1):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, Integer(0))
            else:
                ex2 = ex2.subs(a, Float(round(a, 3), 3))
    return ex2
