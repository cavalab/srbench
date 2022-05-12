from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify, srepr, nsimplify, N

def simplify_fn(equation):
    python_eq = parse_expr(equation, evaluate=False)
    simplified = N(nsimplify(simplify(python_eq), tolerance=1e-8, rational=True))
    return srepr(simplified)