import pdb
import pandas as pd
from yaml import load, Loader
import sympy
# from sympy import *
from sympy import Symbol, simplify, factor, Float, preorder_traversal, Integer
from sympy.parsing.sympy_parser import parse_expr
import re
import ast 


def complexity(expr):
    c=0
    for arg in preorder_traversal(expr):
        c += 1
    return c
        
def round_floats(ex1):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a,Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, 3))
    return ex2

