"""
Simplification code for reducing the number of constants in a mathematical
expression
"""

from collections import defaultdict
from itertools import combinations

from ..operator_definitions \
    import CONSTANT, INTEGER, VARIABLE, MULTIPLICATION, ADDITION
from .expression import Expression


def fold_constants(expression):
    """
    The goal of this function is to combine the constants (and derivative
    functions from only constant values) in an expression into as few constants
    as possible.

    Parameters
    ----------
    expression: `Expression`
        The expression in which the constants will be folded

    Returns
    -------
        A new expression containing potentially fewer constants
    """
    expression = _group_constants(expression)

    check_for_folding = True
    while check_for_folding:
        check_for_folding = False
        constants = _get_constants(expression)
        for const_subset in _subsets(list(constants)):
            insertion_points = _find_insertion_points(expression, const_subset)
            replacements = _generate_replacement_instructions(const_subset,
                                                              constants,
                                                              insertion_points)
            if len(replacements) > 0:
                expression = _perform_constant_folding(expression,
                                                       replacements)
                check_for_folding = True
                break

    return expression


def _group_constants(expression):
    if expression.operator in [CONSTANT, INTEGER, VARIABLE]:
        return expression

    new_operands = [_group_constants(operand)
                    for operand in expression.operands]

    if expression.operator in [MULTIPLICATION, ADDITION]:
        const_operands = [operand for operand in new_operands
                          if operand.is_constant_valued]
        non_const_operands = [operand for operand in new_operands
                              if not operand.is_constant_valued]
        if len(const_operands) > 1 and len(non_const_operands) > 0:
            const_expr = Expression(expression.operator, const_operands)
            return Expression(expression.operator,
                              [const_expr] + non_const_operands)

    return Expression(expression.operator, new_operands)


def _generate_replacement_instructions(const_subset, constants,
                                       insertion_points):
    if len(insertion_points) > len(const_subset):
        return {}

    replacements = defaultdict(dict)
    constants_to_insert = set()
    expressions_to_replace = set()
    for const_num, (_, insertions) in zip(const_subset,
                                          insertion_points.items()):
        const_to_insert = constants[const_num]
        for (parent, children) in insertions:
            for i, child in enumerate(children):
                expressions_to_replace.add(child)
                if i == 0:
                    replacements[parent][child] = const_to_insert
                    constants_to_insert.add(const_to_insert)
                else:
                    replacements[parent][child] = None
                    constants_to_insert.add(None)

    if constants_to_insert == expressions_to_replace:
        return {}
    return replacements


def _get_constants(expression):
    if expression.operator == CONSTANT:
        return {expression.operands[0]: expression}

    if expression.operator in [INTEGER, VARIABLE]:
        return {}

    constants = {}
    for operand in expression.operands:
        constants.update(_get_constants(operand))
    return constants


def _subsets(constants):
    for i in range(1, len(constants) + 1):
        for comb in combinations(constants, i):
            yield set(comb)


def _find_insertion_points(expression, constants):
    if not expression.depends_on.isdisjoint(constants) and \
            len(expression.depends_on - constants - {"i"}) == 0:
        return {expression: [(None, frozenset([expression]))]}

    insertion_points = defaultdict(set)
    _recursive_insertion_point_search(expression, constants, insertion_points,
                                      parent=None)
    return insertion_points


def _recursive_insertion_point_search(expression, constants, insertion_points,
                                      parent):
    if expression.operator in [CONSTANT, INTEGER, VARIABLE]:
        return

    for operand in expression.operands:
        _recursive_insertion_point_search(operand, constants, insertion_points,
                                          expression)

    if not _is_insertion_point_for_constants(expression, constants):
        return

    if expression.is_constant_valued:
        insertion_points[expression].add((parent, frozenset([expression])))
    else:
        constant_operands = \
            frozenset([operand for operand in expression.operands
                       if len(operand.depends_on - constants - {"i"}) == 0])
        insertion_points[expression].add((expression, constant_operands))

    return


def _is_insertion_point_for_constants(expression, constants):
    solely_const_based = []
    has_others = []
    for operand in expression.operands:
        has_consts = not operand.depends_on.isdisjoint(constants)
        has_others.append(len(operand.depends_on - constants - {"i"}) > 0)
        solely_const_based.append(has_consts and not has_others[-1])
    return any(solely_const_based) and any(has_others)


def _perform_constant_folding(expression, replacements):
    if None in replacements:
        return replacements[None][expression].copy()

    return _recursive_expreson_replacement(expression, replacements)


def _recursive_expreson_replacement(expression, replacements):
    if expression not in replacements:
        if expression.operator in [CONSTANT, INTEGER, VARIABLE]:
            return expression
        return expression.map(lambda x:
                              _perform_constant_folding(x, replacements))
    new_operands = _get_new_operands_with_replacements(expression,
                                                       replacements)
    return Expression(expression.operator, new_operands)


def _get_new_operands_with_replacements(expression, replacements):
    new_operands = []
    for operand in expression.operands:
        replacements_for_expr = replacements[expression]
        if operand in replacements_for_expr:
            operand_replacement = replacements_for_expr[operand]
            if operand_replacement is not None:
                new_operands.append(operand_replacement.copy())
        else:
            new_operands.append(_perform_constant_folding(operand,
                                                          replacements))
    return new_operands
