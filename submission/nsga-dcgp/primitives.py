import sympy as sp
import torch


class Function:
    def __init__(self, pt_func, sp_func, arity):
        self.pt_func = pt_func
        self.sp_func = sp_func
        self.arity = arity

    def __call__(self, *args):
        return self.pt_func(*args)

    def expr(self, *args):
        variables = list([sp.Symbol(arg) if isinstance(arg, str) else arg for arg in args])
        return self.sp_func(*variables)


def _sp_square(*args):
    return args[0] ** 2


def _sp_sub(*args):
    return args[0] - args[1]


def _sp_div(*args):
    return args[0] / args[1]


sqrt = Function(torch.sqrt, sp.sqrt, 1)
square = Function(torch.square, _sp_square, 1)
log = Function(torch.log, sp.log, 1)
sin = Function(torch.sin, sp.sin, 1)
cos = Function(torch.cos, sp.sin, 1)
tan = Function(torch.tan, sp.tan, 1)
abs = Function(torch.abs, sp.Abs, 1)
exp = Function(torch.exp, sp.exp, 1)

add = Function(torch.add, sp.Add, 2)
sub = Function(torch.sub, _sp_sub, 2)
mul = Function(torch.mul, sp.Mul, 2)
div = Function(torch.div, _sp_div, 2)

primitive_map = {
    'sqrt': sqrt,
    'square': square,
    'log': log,
    'sin': sin,
    'cos': cos,
    'tan': tan,
    'abs': abs,
    'exp': exp,

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
