"""
Definition of a mathematical expression in tree form that can be used for
simplification tasks.
"""
from ..operator_definitions \
    import POWER, INTEGER, MULTIPLICATION, CONSTANT, VARIABLE, ADDITION


class Expression:
    """A mathematical expression

    Parameters
    ----------
    operator: ENUM from operator definitions
        The mathematical operator in the expression.
    operands: list of `Expression` or int
        The operands upon which the operator in the expression acts.
    """
    def __init__(self, operator, operands):
        self._operator = operator
        self._operands = operands
        self._is_constant_valued = None
        self._depends_on = None
        self._hash = None

    @property
    def operator(self):
        """The primary mathematical operation in the expression"""
        return self._operator

    @property
    def operands(self):
        """The operands upon which the expressions operator acts"""
        return self._operands

    @property
    def is_constant_valued(self):
        """is the expression derived from solely constant values"""
        if self._is_constant_valued is None:
            self._is_constant_valued = self._is_derived_from_constants()
        return self._is_constant_valued

    @property
    def depends_on(self):
        """The constants, integers and variables in subexpressions"""
        if self._depends_on is None:
            self._depends_on = self._find_what_expression_depends_on()
        return self._depends_on

    @property
    def base(self):
        """The base x in x^b"""
        if self._operator == POWER:
            return self._operands[0]
        if self._operator == INTEGER:
            return None
        return self

    @property
    def exponent(self):
        """The exponent b in x^b"""
        if self._operator == POWER:
            return self._operands[1]
        if self._operator == INTEGER:
            return None
        return Expression(INTEGER, [1, ])

    @property
    def term(self):
        """The term x in A*x"""
        if self._operator == MULTIPLICATION:
            if self._operands[0].operator in [INTEGER, CONSTANT]:
                return Expression(MULTIPLICATION, self._operands[1:])
            return self

        if self._operator == INTEGER:
            return None
        return Expression(MULTIPLICATION, [self, ])

    @property
    def coefficient(self):
        """The coefficient A in A*x"""
        if self._operator == MULTIPLICATION and \
                self._operands[0].operator in [INTEGER, CONSTANT]:
            return self._operands[0]
        if self._operator == INTEGER:
            return None
        return Expression(INTEGER, [1, ])

    def _is_derived_from_constants(self):
        if self._operator in [INTEGER, CONSTANT]:
            return True

        if self._operator == VARIABLE:
            return False

        for operand in self._operands:
            if not operand.is_constant_valued:
                return False

        return True

    def _find_what_expression_depends_on(self):
        if self._operator == INTEGER:
            return {"i"}
        if self._operator == VARIABLE:
            return {"x"}
        if self._operator == CONSTANT:
            return {self._operands[0]}

        return set.union(*[o.depends_on for o in self._operands])

    def map(self, mapped_function):
        """ Apply a function to all operands of an expression

        Parameters
        ----------
        mapped_function: callable
            The function to be applied to the operands

        Returns
        -------
            A new expression with mapped operands
        """
        mapped_operands = [mapped_function(i) for i in self._operands]
        return Expression(self._operator, mapped_operands)

    def is_zero(self):
        """is the expression == 0"""
        if self._operator != INTEGER:
            return False
        return self._operands[0] == 0

    def is_one(self):
        """is the expression == 1"""
        if self._operator != INTEGER:
            return False
        return self._operands[0] == 1

    def __eq__(self, other):
        if other is None:
            return False
        if self._operator != other.operator:
            return False
        return self._operands == other.operands

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.is_constant_valued or other.is_constant_valued:
            return self._constant_lt(other)

        s_op = self._operator
        o_op = other.operator
        if MULTIPLICATION in (s_op, o_op):
            return self._associative_lt(other, MULTIPLICATION)
        if POWER in (s_op, o_op):
            return self._power_lt(other)
        if ADDITION in (s_op, o_op):
            return self._associative_lt(other, ADDITION)
        return self._general_lt(other)

    def _constant_lt(self, other):
        if self.is_constant_valued != other.is_constant_valued:
            return self.is_constant_valued

        return self._general_lt(other)

    def _general_lt(self, other):
        if self._operator != other.operator:
            return self._operator < other.operator
        return self._operands_lt(self._operands, other.operands)

    @staticmethod
    def _operands_lt(s_operands, o_operands):
        for s_operand, o_operand in zip(reversed(s_operands),
                                        reversed(o_operands)):
            if s_operand != o_operand:
                return s_operand < o_operand
        return len(s_operands) < len(o_operands)

    def _associative_lt(self, other, associative_operator):
        if self._operator == associative_operator:
            if other.operator == associative_operator:
                return self._operands_lt(self._operands,
                                         other.operands)
            return self._operands_lt(self._operands, [other, ])
        return self._operands_lt([self, ], other.operands)

    def _power_lt(self, other):
        if self._operator == POWER:
            s_base = self._operands[0]
            s_exponent = self._operands[1]
        else:
            s_base = self
            s_exponent = Expression(INTEGER, [1])

        if other.operator == POWER:
            o_base = other.operands[0]
            o_exponent = other.operands[1]
        else:
            o_base = other
            o_exponent = Expression(INTEGER, [1])

        if s_base == o_base:
            return s_exponent < o_exponent
        return s_base < o_base

    def copy(self):
        """Shallow copy of the expression"""
        return Expression(self._operator, self._operands)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = f"{self._operator}("
        for operand in self._operands:
            string += f"{operand}, "
        string += ")"
        return string

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self._operator,) +
                              tuple(hash(i) for i in self._operands))
        return self._hash
