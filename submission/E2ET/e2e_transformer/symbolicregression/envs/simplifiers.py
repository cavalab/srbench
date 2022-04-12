import traceback
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .generators import all_operators, math_constants, Node, NodeList
from sympy.core.rules import Transform
import numpy as np
from functools import partial
import numexpr as ne
import sympytorch
import torch
from ..utils import timeout, MyTimeoutError

def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0
    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                return f
            else:
                return f2
        except MyTimeoutError:
            return f
        except Exception as e:
            return f
    return _simplify(f)

class InvalidPrefixExpression(BaseException):
    pass

import signal
from contextlib import contextmanager

@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError

class Simplifier:
    def __init__(self, generator):

        self.params = generator.params
        self.encoder = generator.equation_encoder
        self.operators = generator.operators
        self.max_int = generator.max_int
        self.local_dict = {
            "n": sp.Symbol("n", real=True, nonzero=True, positive=True, integer=True),
            "e": sp.E,
            "pi": sp.pi,
            "euler_gamma": sp.EulerGamma,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "step": sp.Heaviside,
            "sign": sp.sign,
        }
        for k in generator.variables:
            self.local_dict[k] = sp.Symbol(k, real=True, integer=False)
        
    def expand_expr(self, expr):
        with timeout(1):
            expr = sp.expand(expr)
        return expr

    def simplify_expr(self, expr):
        with timeout(1):
            expr = sp.simplify(expr)
        return expr

    def tree_to_sympy_expr(self, tree):
        prefix = tree.prefix().split(",")
        sympy_compatible_infix = self.prefix_to_sympy_compatible_infix(prefix)
        expr = parse_expr(sympy_compatible_infix, evaluate=True, local_dict=self.local_dict)
        return expr

    def tree_to_torch_module(self, tree, dtype=torch.float32):
        expr = self.tree_to_sympy_expr(tree)
        mod = self.expr_to_torch_module(expr, dtype)
        return mod

    def expr_to_torch_module(self, expr, dtype):
        mod = sympytorch.SymPyModule(expressions=[expr])
        mod.to(dtype)
        def wrapper_fn(_mod, x, constants=None):
            local_dict = {}
            for d in range(x.shape[1]):
                local_dict["x_{}".format(d)]=x[:, d]
            if constants is not None:
                for d in range(constants.shape[0]):
                    local_dict["CONSTANT_{}".format(d)]=constants[d]
            return _mod(**local_dict)
        return partial(wrapper_fn, mod)

    def expr_to_numpy_fn(self, expr):

        def wrapper_fn(_expr, x, extra_local_dict={}):
            local_dict = {}
            for d in range(x.shape[1]):
                local_dict["x_{}".format(d)]=x[:, d]
            local_dict.update(extra_local_dict)
            variables_symbols = sp.symbols(' '.join(["x_{}".format(d) for d in range(x.shape[1])]))
            extra_symbols = list(extra_local_dict.keys())
            if len(extra_symbols)>0:
                extra_symbols = sp.symbols(' '.join(extra_symbols))
            else:
                extra_symbols=()
            np_fn =  sp.lambdify((*variables_symbols, *extra_symbols), _expr, modules='numpy')
            return np_fn(**local_dict)

        return partial(wrapper_fn, expr)

    def tree_to_numpy_fn(self, tree):
        expr = self.tree_to_sympy_expr(tree)
        return self.expr_to_numpy_fn(expr)

    def tree_to_numexpr_fn(self, tree):
        infix = tree.infix()
        numexpr_equivalence = {
                                "add": "+",
                                "sub": "-",
                                "mul": "*",
                                "pow": "**",
                                "inv": "1/",
        }

        for old, new in numexpr_equivalence.items():
            infix=infix.replace(old, new)

        def get_vals(dim, val):
            vals_ar = np.empty((dim,))
            vals_ar[:] = val
            return vals_ar

        def wrapped_numexpr_fn(_infix, x, extra_local_dict={}):
            assert isinstance(x,  np.ndarray) and len(x.shape)==2
            local_dict = {}
            for d in range(self.params.max_input_dimension):
                if "x_{}".format(d) in _infix:
                    if d >= x.shape[1]: 
                        local_dict["x_{}".format(d)]=np.zeros(x.shape[0])
                    else:
                        local_dict["x_{}".format(d)]=x[:,d]
            local_dict.update(extra_local_dict)
            try:
                vals = ne.evaluate(_infix, local_dict=local_dict)
                if len(vals.shape)==0:
                    vals = get_vals(x.shape[0], vals)
            except Exception as e:
                print(e)
                print("problem with tree", _infix)
                traceback.format_exc()
                vals = get_vals(x.shape[0], np.nan)
            return vals[:, None]
        return partial(wrapped_numexpr_fn, infix)

    def sympy_expr_to_tree(self, expr):
        prefix = self.sympy_to_prefix(expr)
        return self.encoder.decode(prefix)

    def round_expr(self, expr, decimals=4):
        with timeout(1):
            expr = expr.xreplace(Transform(lambda x: x.round(decimals), lambda x: isinstance(x, sp.Float)))
        return expr
        
    def float_to_int_expr(self, expr):
        floats = expr.atoms(sp.Float)
        ints = [fl for fl in floats if int(fl)==fl]
        expr = expr.xreplace(dict(zip(ints, [int(i) for i in ints])))
        return expr
        
    def apply_fn(self, tree, fn_stack=[]):
        expr = self.tree_to_sympy_expr(tree)
        for (fn, arg) in fn_stack:
            expr = getattr(self, fn)(expr=expr, **arg)   
        new_tree =  self.sympy_expr_to_tree(expr)
        if new_tree is None:
            new_tree = tree
        return new_tree

    def write_infix(self, token, args):
        """
        Infix representation.
    
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        if token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "idiv":
            return f"idiv({args[0]},{args[1]})"
        elif token == "mod":
            return f"({args[0]})%({args[1]})"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token in all_operators:
            return f"{token}({args[0]})"
        else:
            return token
        raise InvalidPrefixExpression(
            f"Unknown token in prefix expression: {token}, with arguments {args}"
        )

    def _prefix_to_sympy_compatible_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in all_operators:
            args = []
            l1 = expr[1:]
            for _ in range(all_operators[t]):
                i1, l1 = self._prefix_to_sympy_compatible_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        else:  # leaf
            try:
                float(t)
                t = str(t)
            except ValueError:
                t=t
            return t, expr[1:]

    def prefix_to_sympy_compatible_infix(self, expr):
        """
        Convert prefix expressions to a format that SymPy can parse.
        """
        p, r = self._prefix_to_sympy_compatible_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        # if op == 'pow':
        #     if isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
        #         return ['sqrt'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='2':
        #         return ['sqr'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-1':
        #         return ['inv'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-2':
        #         return ['inv', 'sqr'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Float):
            s = str(expr)
            return [s]
        elif isinstance(expr, sp.Rational):
            return ["mul", str(expr.p), "pow", str(expr.q), "-1"]
        elif expr == sp.EulerGamma:
            return ["euler_gamma"]
        elif expr == sp.E:
            return ["e"]
        elif expr == sp.pi:
            return ["pi"]

        # if we want div and sub
        # if isinstance(expr, sp.Mul) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and isinstance(expr.args[0].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and isinstance(expr.args[1].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])
        # if isinstance(expr, sp.Add) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and str(expr.args[0].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and str(expr.args[1].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])

        # if isinstance(expr, sp.Pow) and str(expr.args[1])=='-1':
        #     return ['inv'] + self.sympy_to_prefix(expr.args[0])

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)

        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Mod: "mod",
        sp.Pow: "pow",
        # Misc
        sp.Abs: "abs",
        sp.sign: "sign",
        sp.Heaviside: "step",
        # Exp functions
        sp.exp: "exp",
        sp.log: "log",
        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        # Trigonometric Inverses
        sp.asin: "arcsin",
        sp.acos: "arccos",
        sp.atan: "arctan",
    }