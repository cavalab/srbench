import numpy as np
import sympy as sp
import torch


def reduce_adder(l1, l2):
    return l1 + l2


def simplicity(indiv):
    eq = sp.sympify(indiv.expr())
    s = eq.count_ops()
    return round(30 / (30 + s), 1)


def accuracy(indiv, X, y):
    with torch.no_grad():
        y_pred = indiv(X)
    y_pred = y_pred.detach()
    sse = ((y_pred - y) ** 2).sum()
    var = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / var
    if not torch.isfinite(r2):
        return -np.inf
    return r2.item()


def dominate(obj_p, obj_q):
    # here, greater is better
    obj_p, obj_q = np.array(obj_p), np.array(obj_q)
    return (obj_p >= obj_q).all() and (obj_p > obj_q).any()


def print_info(gen, indiv):
    print("Gen {}: Best EQ={} || R2Score={}, Simplicity={}".format(gen, indiv.expr(), indiv.fitness, indiv.simplicity))


# def crowding_distance(pop, obj_names):
#     for i in range(len(pop)):
#         pop[i].crowding_distance = 0
#
#     for obj_name in obj_names:
#         pop = sorted(pop, key=lambda indiv: getattr(indiv, obj_name), reverse=True)
