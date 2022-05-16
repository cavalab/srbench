"""
Simplification operations that are not strictly necessary, but could be
performed to maintain a specific cannonical structure.

The module constants below control this behavior:

INSERT_SUBTRACTION: Default True.
    a + (-1)*b is converted to a - b

REPLACE_INTEGER_POWERS: Default True.
    a^5 is converted to (a*a)*(a*a)*a

REPLACE_INTEGERS_WITH_CONSTANTS: Default False
    x+x is simplified to 2*x and is converted to c*x
"""
from ..operator_definitions import INTEGER, CONSTANT, VARIABLE, ADDITION, \
                                   MULTIPLICATION, SUBTRACTION, POWER
from .expression import Expression

INSERT_SUBTRACTION = True
REPLACE_INTEGER_POWERS = True
REPLACE_INTEGERS_WITH_CONSTANTS = False

NEGATIVE_ONE = Expression(INTEGER, [-1])
SOME_BIG_INT = 1000000


def optional_modifications(expression):
    """Modification of expression using non-essential heuristics

    INSERT_SUBTRACTION: Default on
        a + (-1)*b is converted to a - b

    REPLACE_INTEGER_POWERS: Default on
        a^5 is converted to (a*a)*(a*a)*a

    REPLACE_INTEGERS_WITH_CONSTANTS: Default off
        x+x is simplified to 2*x and is converted to c*x

    Parameters
    ----------
    expression: `Expression`
        The expression to modify

    Returns
    -------
        expression with designated simplifications made
    """
    if INSERT_SUBTRACTION:
        expression = _insert_subtraction(expression)
    if REPLACE_INTEGER_POWERS:
        expression = _replace_integer_powers(expression)
    if REPLACE_INTEGERS_WITH_CONSTANTS:
        expression = _replace_integers_with_constants(expression)
    return expression


def _insert_subtraction(expression):
    operator = expression.operator
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_subtraction = [_insert_subtraction(operand)
                              for operand in expression.operands]
    if operator != ADDITION:
        return Expression(operator, operands_w_subtraction)

    additive_operands = []
    subtractive_operands = []
    for operand in operands_w_subtraction:
        if operand.coefficient == NEGATIVE_ONE:
            term = operand.term
            if len(term.operands) == 1:
                subtractive_operands.append(term.operands[0])
            else:
                subtractive_operands.append(term)
        else:
            additive_operands.append(operand)

    if len(subtractive_operands) == 0:
        return Expression(ADDITION, additive_operands)

    if len(additive_operands) == 0:
        return Expression(MULTIPLICATION, [NEGATIVE_ONE.copy(),
                                           Expression(ADDITION,
                                                      subtractive_operands)])

    if len(subtractive_operands) == 1:
        subtractive_exp = subtractive_operands[0]
    else:
        subtractive_exp = Expression(ADDITION, subtractive_operands)

    if len(additive_operands) == 1:
        additive_exp = additive_operands[0]
    else:
        additive_exp = Expression(ADDITION, additive_operands)

    return Expression(SUBTRACTION, [additive_exp, subtractive_exp])


def _replace_integer_powers(expression):
    operator = expression.operator
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_replaced = [_replace_integer_powers(operand)
                           for operand in expression.operands]

    if operator != POWER or operands_w_replaced[1].operator != INTEGER \
            or operands_w_replaced[1].operands[0] <= 0:
        return Expression(operator, operands_w_replaced)

    power = operands_w_replaced[1].operands[0]
    return Expression(MULTIPLICATION, [operands_w_replaced[0]] * power)


def _replace_integers_with_constants(expression):
    operator = expression.operator
    if operator in [CONSTANT, VARIABLE]:
        return expression
    if operator == INTEGER:
        return Expression(CONSTANT, [SOME_BIG_INT + expression.operands[0]])

    operands_w_replaced = [_replace_integers_with_constants(operand)
                           for operand in expression.operands]
    return Expression(operator, operands_w_replaced)
