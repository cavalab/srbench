"""
Bingo's simplification algorithm using a built-in computer algebra system
"""
from .constant_folding import fold_constants
from .automatic_simplification import automatic_simplify
from .optional_expression_modification import optional_modifications
from .interpreter import build_cas_expression, build_agraph_stack


def simplify(agraph_stack):
    """
    Simplification of an equation (represented by an AGraph command array)
    using bingo's built-in computer algebra system

    Parameters
    ----------
    agraph_stack: AGraph command array
        An equation to be simplified

    Returns
    -------
        The simplified equation
    """
    cas_expression = build_cas_expression(agraph_stack)

    cas_expression = automatic_simplify(cas_expression)
    cas_expression = fold_constants(cas_expression)

    cas_expression = optional_modifications(cas_expression)

    return build_agraph_stack(cas_expression)
