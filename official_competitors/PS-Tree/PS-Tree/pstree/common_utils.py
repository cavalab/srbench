import random

import numpy as np
from deap.gp import Primitive


def reset_random(s):
    random.seed(s)
    np.random.seed(s)


infix_map = {
    'add_2': '+',
    'sub_2': '-',
    'multiply': '*',
    'protect_division': '/',
}


def gene_to_string(gene):
    string = ""
    stack = []
    for node in gene:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            if type(prim) is Primitive:
                string = '('
                if prim.name == 'analytical_quotient':
                    string += f'{args[0]}/sqrt(1+{args[1]}*{args[1]})'
                elif prim.name == 'analytical_loge':
                    string += f'log(1+Abs({args[0]}))'
                elif prim.name == 'protected_sqrt':
                    string += f'sqrt(Abs({args[0]}))'
                elif prim.name == 'maximum':
                    string += f'Max({args[0]}, {args[1]})'
                elif prim.name == 'minimum':
                    string += f'Min({args[0]}, {args[1]})'
                elif prim.name == 'negative':
                    string += f'-{args[0]}'
                elif prim.name not in infix_map:
                    string += prim.format(*args)
                else:
                    string += args[0]
                    for a in args[1:]:
                        string += f'{infix_map[prim.name]}{a}'
                string += ')'
            else:
                string = prim.name
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string
