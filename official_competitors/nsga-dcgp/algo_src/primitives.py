from functools import partial

import torch


class Function:
    def __init__(self, pt_func, sp_func, arity):
        self.pt_func = pt_func
        self.sp_func = sp_func
        self.arity = arity

    def __call__(self, *args):
        return self.pt_func(*args)

    def expr(self, *args):
        return self.sp_func(*args)


def _sp_add(*args):
    return f"Add({args[0]}, {args[1]})"


def _sp_sub(*args):
    return f"Add({args[0]}, -{args[1]})"


def _sp_mul(*args):
    return f"Mul({args[0]}, {args[1]})"


def _sp_div(*args):
    return f"Mul({args[0]}, 1/{args[1]})"


def _unary_sp_oper(*args, op=''):
    return f"{op}({args[0]})"


def _sp_log(*args):
    return f"log(Abs({args[0]}))"


def _sp_sqrt(*args):
    return f"sqrt(Abs({args[0]}))"


def _sp_power(*args, base=None):
    if base:
        return f"({args[0]})**{str(base)}"
    return f"{args[0]}"


def _protected_log(*args):
    return torch.log(torch.abs(args[0]))


def _protected_sqrt(*args):
    return torch.sqrt(torch.abs(args[0]))


sqrt = Function(_protected_sqrt, _sp_sqrt, 1)
square = Function(torch.square, partial(_sp_power, base=2), 1)
cube = Function(partial(torch.pow, exponent=3), partial(_sp_power, base=3), 1)
log = Function(_protected_log, _sp_log, 1)
sin = Function(torch.sin, partial(_unary_sp_oper, op='sin'), 1)
cos = Function(torch.cos, partial(_unary_sp_oper, op='cos'), 1)
# tan = Function(torch.tan, sp.tan, 1)
abs = Function(torch.abs, partial(_unary_sp_oper, op='Abs'), 1)
# exp = Function(torch.exp, sp.exp, 1)

add = Function(torch.add, _sp_add, 2)
sub = Function(torch.sub, _sp_sub, 2)
mul = Function(torch.mul, _sp_mul, 2)
div = Function(torch.div, _sp_div, 2)

primitive_map = {
    'sqrt': sqrt,
    'square': square,
    'cube': cube,
    'log': log,
    'sin': sin,
    'cos': cos,
    # 'tan': tan,
    'abs': abs,
    # 'exp': exp,

    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div
}

default_primitives = list(primitive_map.keys())


def create_functions(primitives):
    if isinstance(primitives[0], Function):
        return primitives
    return list(primitive_map[primitive] for primitive in primitives)


def max_arity(functions):
    ma = 0
    for func in functions:
       ma = max(func.arity, ma)
    return ma
