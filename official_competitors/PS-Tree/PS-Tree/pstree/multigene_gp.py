import copy
import random
from functools import wraps

import numpy as np
from deap.gp import PrimitiveTree, compile, cxOnePoint, mutUniform
from scipy.special import softmax


class MultipleGeneGP():
    def __init__(self, content, gene_num):
        self.gene = []
        self.gene_num = gene_num
        for i in range(self.gene_num):
            self.gene.append(PrimitiveTree(content()))

    def random_select(self):
        return self.gene[random.randint(0, self.gene_num - 1)]

    def weight_select(self):
        weight = np.abs(self.coef)[:, :-1].mean(axis=0)
        p = softmax(-abs(weight))
        return self.gene[np.random.choice(np.arange(len(weight)), p=p)]

    def deterministic_select(self):
        weight = np.abs(self.coef)[:, :-1].mean(axis=0)
        return self.gene[np.argmax(-weight)]

    def __len__(self):
        return sum([len(g) for g in self.gene])


def multiple_gene_evaluation(compiled_genes, x):
    result = []
    for gene in compiled_genes:
        result.append(gene(*x))
    return result


def multiple_gene_initialization(container, generator, gene_num=5):
    return container(generator, gene_num)


def multiple_gene_compile(expr: MultipleGeneGP, pset):
    gene_compiled = []
    for gene in expr.gene:
        gene_compiled.append(compile(gene, pset))
    return gene_compiled


def cxOnePoint_multiple_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.random_select(), ind2.random_select())
    return ind1, ind2


def mutUniform_multiple_gene(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.random_select(), expr, pset)
    return individual,


def cxOnePoint_multiple_gene_weight(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.weight_select(), ind2.weight_select())
    return ind1, ind2


def mutUniform_multiple_gene_weight(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.weight_select(), expr, pset)
    return individual,


def cxOnePoint_multiple_gene_deterministic(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.deterministic_select(), ind2.deterministic_select())
    return ind1, ind2


def mutUniform_multiple_gene_deterministic(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.deterministic_select(), expr, pset)
    return individual,


def staticLimit_multiple_gene(key, max_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                limit_exceed = False
                for x in ind.gene:
                    if key(x) > max_value:
                        limit_exceed = True
                        break
                if limit_exceed:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator


def result_calculation(func, data):
    result = multiple_gene_evaluation(func, data.T)
    for i in range(len(result)):
        yp = result[i]
        if not isinstance(yp, np.ndarray):
            yp = np.full(len(data), 0)
        elif yp.size == 1:
            yp = np.full(len(data), yp)
        result[i] = yp
    result = np.concatenate([np.array(result).T, np.ones((len(data), 1))], axis=1)
    return result
