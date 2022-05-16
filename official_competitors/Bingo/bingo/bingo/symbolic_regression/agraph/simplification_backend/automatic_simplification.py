"""
The core of the built in computer algebra simplification algorithm.

This algorithm is based on the algorithm presented in chapter 3 of Joel Cohen's
book [1].

... [1] Joel S. Cohen (2003) Computer Algebra and Symbolic Computation
"""

from ..operator_definitions import CONSTANT, INTEGER, VARIABLE, POWER, \
                                   MULTIPLICATION, ADDITION, SUBTRACTION, \
                                   DIVISION, SIN, COS, LOGARITHM, EXPONENTIAL, \
                                   ABS, SQRT, SAFE_POWER
from .expression import Expression


NEGATIVE_ONE = Expression(INTEGER, [-1])
ZERO = Expression(INTEGER, [0])
ONE = Expression(INTEGER, [1])


def automatic_simplify(expression):
    """A recursive simplification of an expression

    Parameters
    ----------
    expression: `Expression`
        an expression to be recursively simplified

    Returns
    -------
        a simplified `Expression`
    """
    if expression.operator in [CONSTANT, INTEGER, VARIABLE]:
        return expression

    expr_w_simp_operands = expression.map(automatic_simplify)

    return SIMPLIFICATION_FUNCTIONS[expr_w_simp_operands.operator](
            expr_w_simp_operands)


def simplify_power(expression):
    """simplification of power operators"""
    base, exponent = expression.operands
    if base.is_one():
        return ONE.copy()
    if base.is_zero() and exponent.operator == INTEGER \
            and exponent.operands[0] > 0:
        return ZERO.copy()
    if exponent.operator in [INTEGER, CONSTANT]:
        return _simplify_constant_power(base, exponent)
    return expression


def _simplify_constant_power(base, exponent):
    if exponent.is_one():
        return base
    if exponent.is_zero():
        return ONE.copy()

    if base.operator == INTEGER and exponent.operator == INTEGER \
            and exponent.operands[0] > 0:
        return Expression(INTEGER, [base.operands[0]**exponent.operands[0]])

    if base.operator == POWER:  # multiply constant powers
        base_base = base.operands[0]
        base_exponent = base.operands[1]
        mult_exp = Expression(MULTIPLICATION, [base_exponent, exponent])
        new_exponent = simplify_product(mult_exp)
        if base_exponent.operator in [INTEGER, CONSTANT]:
            return _simplify_constant_power(base_base, new_exponent)
        return Expression(POWER, [base_base, new_exponent])

    if base.operator == MULTIPLICATION:  # distribute constant powers
        def temp_simp_const_power(bas):
            exp = exponent.copy()
            return _simplify_constant_power(bas, exp)
        return simplify_product(base.map(temp_simp_const_power))

    return Expression(POWER, [base, exponent])


def simplify_product(expression):
    """simplification of multiplication operators"""
    operands = expression.operands
    if ZERO in operands:
        return ZERO.copy()
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_product_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ONE.copy()
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return Expression(MULTIPLICATION, recursively_simplified_operands)


def _simplify_product_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] * op_2.operands[0]
            simpl_const_prod = Expression(INTEGER, [new_integer])
            if simpl_const_prod.is_one():
                return []
            return [simpl_const_prod]

        if MULTIPLICATION not in (op_1.operator, op_2.operator):
            if op_1.is_one():
                return [op_2]
            if op_2.is_one():
                return [op_1]

            if op_1.base == op_2.base:
                new_exponent = Expression(ADDITION,
                                          [op_1.exponent, op_2.exponent])
                new_exponent = simplify_sum(new_exponent)
                combined_op = Expression(POWER, [op_1.base, new_exponent])
                combined_op = simplify_power(combined_op)

                if combined_op.is_one():
                    return []
                return [combined_op]

            if op_2 < op_1:
                return [op_2, op_1]

            return operands

        if op_1.operator == MULTIPLICATION:
            to_merge_1 = op_1.operands
        else:
            to_merge_1 = [op_1]
        if op_2.operator == MULTIPLICATION:
            to_merge_2 = op_2.operands
        else:
            to_merge_2 = [op_2]
        return _merge_products(to_merge_1, to_merge_2)

    rest_simplified = _simplify_product_rec(operands[1:])
    if operands[0].operator == MULTIPLICATION:
        return _merge_products(operands[0].operands, rest_simplified)
    return _merge_products([operands[0]], rest_simplified)


def _merge_products(operands_1, operands_2):
    if len(operands_1) == 0:
        return operands_2
    if len(operands_2) == 0:
        return operands_1

    simplified_firsts = _simplify_product_rec([operands_1[0], operands_2[0]])
    if len(simplified_firsts) == 0:
        return _merge_products(operands_1[1:], operands_2[1:])
    if len(simplified_firsts) == 1:
        return simplified_firsts + _merge_products(operands_1[1:],
                                                   operands_2[1:])
    if simplified_firsts[0] == operands_1[0]:
        return [simplified_firsts[0]] + _merge_products(operands_1[1:],
                                                        operands_2)
    return [simplified_firsts[0]] + _merge_products(operands_1, operands_2[1:])


def simplify_sum(expression):
    """simplification of addition operators"""
    operands = expression.operands
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_sum_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ZERO.copy()
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return Expression(ADDITION, recursively_simplified_operands)


def _simplify_sum_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] + op_2.operands[0]
            simpl_const_sum = Expression(INTEGER, [new_integer])
            if simpl_const_sum.is_zero():
                return []
            return [simpl_const_sum]

        if ADDITION not in (op_1.operator, op_2.operator):
            if op_1.is_zero():
                return [op_2]
            if op_2.is_zero():
                return [op_1]

            if op_1.term == op_2.term:
                new_coefficient = Expression(ADDITION,
                                             [op_1.coefficient,
                                              op_2.coefficient])
                new_coefficient = simplify_sum(new_coefficient)
                combined_op = Expression(MULTIPLICATION,
                                         [new_coefficient, op_1.term])
                combined_op = simplify_product(combined_op)

                if combined_op.is_zero():
                    return []
                return [combined_op]

            if op_2 < op_1:
                return [op_2, op_1]

            return operands

        if op_1.operator == ADDITION:
            to_merge_1 = op_1.operands
        else:
            to_merge_1 = [op_1]
        if op_2.operator == ADDITION:
            to_merge_2 = op_2.operands
        else:
            to_merge_2 = [op_2]
        return _merge_sums(to_merge_1, to_merge_2)

    rest_simplified = _simplify_sum_rec(operands[1:])
    if operands[0].operator == ADDITION:
        return _merge_sums(operands[0].operands, rest_simplified)
    return _merge_sums([operands[0]], rest_simplified)


def _merge_sums(operands_1, operands_2):
    if len(operands_1) == 0:
        return operands_2
    if len(operands_2) == 0:
        return operands_1

    simplified_firsts = _simplify_sum_rec([operands_1[0], operands_2[0]])
    if len(simplified_firsts) == 0:
        return _merge_sums(operands_1[1:], operands_2[1:])
    if len(simplified_firsts) == 1:
        return simplified_firsts + _merge_sums(operands_1[1:], operands_2[1:])
    if simplified_firsts[0] == operands_1[0]:
        return [simplified_firsts[0]] + _merge_sums(operands_1[1:], operands_2)
    return [simplified_firsts[0]] + _merge_sums(operands_1, operands_2[1:])


def simplify_quotient(expression):
    """simplification of division operators"""
    numerator, denominator = expression.operands
    denominator_inv = Expression(POWER, [denominator, NEGATIVE_ONE.copy()])
    denominator_inv = simplify_power(denominator_inv)
    quotient_as_product = Expression(MULTIPLICATION,
                                     [numerator, denominator_inv])
    return simplify_product(quotient_as_product)


def simplify_difference(expression):
    """simplification of subtraction operators"""
    first, second = expression.operands
    new_operands = [first]
    if second.operator == ADDITION:
        for operand in second.operands:
            negative_operand = Expression(MULTIPLICATION,
                                          [NEGATIVE_ONE.copy(), operand])
            new_operands.append(simplify_product(negative_operand))
    else:
        negative_second = Expression(MULTIPLICATION,
                                     [NEGATIVE_ONE.copy(), second])
        new_operands.append(simplify_product(negative_second))
    difference_as_sum = Expression(ADDITION, new_operands)
    return simplify_sum(difference_as_sum)


def simplify_sin(expression):
    """simplification of sin operators"""
    if expression.operands[0].is_zero():
        return ZERO.copy()
    return expression


def simplify_cos(expression):
    """simplification of cos operators"""
    if expression.operands[0].is_zero():
        return ONE.copy()
    return expression


def simplify_logarithm(expression):
    """simplification of log operators"""
    operand = expression.operands[0]
    if operand.is_one():
        return ZERO.copy()
    if operand.operator == EXPONENTIAL:
        return operand.operands[0]
    return expression


def simplify_exponential(expression):
    """simplification of exp operators"""
    if expression.operands[0].is_zero():
        return ONE.copy()
    return expression


def no_simplification(expression):
    """no simplification performed"""
    return expression


SIMPLIFICATION_FUNCTIONS = {
    POWER: simplify_power,
    MULTIPLICATION: simplify_product,
    ADDITION: simplify_sum,
    DIVISION: simplify_quotient,
    SUBTRACTION: simplify_difference,
    SIN: simplify_sin,
    COS: simplify_cos,
    LOGARITHM: simplify_logarithm,
    EXPONENTIAL: simplify_exponential,
    ABS: no_simplification,
    SQRT: no_simplification,
    SAFE_POWER: simplify_power,
}
