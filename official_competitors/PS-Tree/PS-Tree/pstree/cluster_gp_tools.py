import builtins
import operator
import random
from collections import Callable
from copy import deepcopy
from math import isclose

from bluepyopt.deapext.tools.selIBEA import _calc_fitness_components, _calc_fitnesses, _environmental_selection
from deap.tools import HallOfFame, selRandom, selTournament
from gplearn.functions import _protected_sqrt
from scipy.special import softmax

from .gp_function import *


def c_deepcopy(a):
    return deepcopy(a)


def analytical_loge(x):
    return np.log(1 + abs(x))


def protected_sqrt(x):
    return np.sqrt(np.abs(x))


def add_pset_function(pset, max_arity, basic_primitive):
    if hasattr(pset, 'add_function'):
        add_function: Callable = pset.add_function
    elif hasattr(pset, 'addPrimitive'):
        add_function = pset.addPrimitive
    else:
        raise Exception("The current type of primitive set is not supported yet!")
    if basic_primitive == 'normalized':
        def normalized_wrapper(func):
            def simple_func(*arg):
                result = func(*arg)
                if np.var(result) == 0:
                    return result
                else:
                    return (result - np.mean(result)) / np.sqrt(np.var(result))

            simple_func.__name__ = f'normalized_{func.__name__}'
            return simple_func

        add_function = lambda func, arity: pset.addPrimitive(normalized_wrapper(func), arity)

    # basic primitives
    add_function(np.multiply, 2)
    add_function(analytical_quotient, 2)

    for i in range(2, max_arity + 1):
        add.__name__ = 'add_{}'.format(i)
        sub.__name__ = 'sub_{}'.format(i)
        add_function(add, i)
        add_function(sub, i)

    # advanced primitives
    if basic_primitive == 'optimal':
        add_function(np.sin, 1)
        add_function(np.tanh, 1)

    if basic_primitive == False or basic_primitive in ['abs-sqrt', 'min-max']:
        add_function(np.sin, 1)
        add_function(np.cos, 1)

    if basic_primitive in ['abs-sqrt', 'min-max']:
        add_function(np.maximum, 2)
        add_function(np.minimum, 2)

    if basic_primitive == 'abs-sqrt':
        add_function(np.abs, 1)
        add_function(_protected_sqrt, 1)

    if isinstance(basic_primitive, str) and ',' in basic_primitive:
        for p in basic_primitive.split(','):
            func = {
                'log': analytical_loge,
                'sqrt': protected_sqrt,
                'sin': np.sin,
                'tanh': np.tanh,
            }[p]
            add_function(func, 1)


class LexicaseHOF(HallOfFame):

    def __init__(self, cluster_gp: bool):
        self.cluster_gp = cluster_gp
        HallOfFame.__init__(self, None)

    def update(self, population):
        """Update the Pareto front hall of fame with the *population* by adding
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        if self.cluster_gp == False:
            for ind in population:
                ind.fitness.values = (np.sum(ind.fitness.values),)
                ind.fitness.wvalues = (np.sum(ind.fitness.wvalues),)

        for p in population:
            self.insert(p)

        hofer_arr: np.ndarray = None
        for i, hofer in enumerate(self):
            if hofer_arr is None:
                hofer_arr = np.array(hofer.fitness.wvalues).reshape(1, -1)
            else:
                hofer_arr = np.concatenate([hofer_arr, np.array(hofer.fitness.wvalues).reshape(1, -1)])
        max_hof = np.max(hofer_arr, axis=0)

        del_ind = []
        max_value = np.full_like(max_hof, -np.inf)
        for index, x in enumerate(self):
            fitness_wvalues = np.array(x.fitness.wvalues)
            if np.any(fitness_wvalues >= max_hof) \
                and np.any(fitness_wvalues > max_value):
                loc = np.where(fitness_wvalues > max_value)
                max_value[loc] = fitness_wvalues[loc]
                continue
            del_ind.append(index)

        for i in reversed(del_ind):
            self.remove(i)


def selDecayRankSum(individuals, k):
    # sum rank + decay
    import pandas as pd
    chosen = []
    cases = list(range(len(individuals[0].fitness.values)))
    fitness_list = []
    for i, ind in enumerate(individuals):
        fitness_list.append(ind.fitness.wvalues)
    fitness = pd.DataFrame(fitness_list).rank(axis=0, ascending=False).values
    for i in range(k):
        random.shuffle(cases)
        for i, f in enumerate(individuals):
            individuals[i].temp_rank = 0
            for j, c in enumerate(cases):
                individuals[i].temp_rank += ((fitness[i][c]) / (1 << j))
        min_individual = builtins.min(individuals, key=operator.attrgetter('temp_rank'))
        chosen.append(random.choice
                      (list(filter(lambda x: isclose(x.temp_rank, min_individual.temp_rank), individuals))))
    return chosen


def selRankSumTournament(individuals, k, tournsize=10):
    # sum rank
    import pandas as pd
    chosen = []
    fitness_list = []
    for i, ind in enumerate(individuals):
        fitness_list.append(ind.fitness.wvalues)
    fitness = pd.DataFrame(fitness_list).rank(axis=0, ascending=False).values.sum(axis=1)
    for i, f in enumerate(fitness):
        individuals[i].rank_sum = f
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        from operator import attrgetter
        chosen.append(builtins.min(aspirants, key=attrgetter('rank_sum')))
    return chosen


class TestFunction():
    def __init__(self):
        self.x = None
        self.y = None
        self.regr = None

    def predict_loss(self):
        if len(self.x) > 0:
            y_p = self.regr.predict(self.x)
            return np.sum((self.y - y_p) ** 2)
        else:
            return 0

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)


def individual_to_tuple(ind):
    arr = []
    for x in ind:
        arr.append(x.name)
    return tuple(arr)


def selTournamentDCDSimple(individuals, k):
    """
    A simplified version of the tournament selection operator based on dominance
    """

    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if random.random() <= 0.5:
            return ind1
        return ind2

    chosen = []
    for i in range(0, k, 2):
        individuals_sample = random.sample(individuals, 4)
        chosen.append(tourn(individuals_sample[0], individuals_sample[1]))
        chosen.append(tourn(individuals_sample[2], individuals_sample[3]))
    return chosen


def remove_duplicate_fitness(candidates):
    # remove duplicated individuals
    final_candidates = []
    fitness_set = set()
    for c in candidates:
        weight = c.fitness.wvalues
        if weight not in fitness_set:
            fitness_set.add(weight)
            final_candidates.append(c)
    return final_candidates


def selAutomaticEpsilonLexicasePlus(individuals, k):
    fit_weights = individuals[0].fitness.weights
    if len(fit_weights) == 1:
        return selTournament(individuals, k, tournsize=5)
    else:
        return selAutomaticEpsilonLexicase(individuals, k)


def selAutomaticEpsilonLexicase(individuals, k):
    selected_individuals = []

    for i in range(k):
        fit_weights = individuals[0].fitness.weights

        candidates = remove_duplicate_fitness(individuals)
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = [x.fitness.values[cases[0]] for x in candidates]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            if fit_weights[cases[0]] > 0:
                best_val_for_case = max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] >= min_val_to_survive])
            else:
                best_val_for_case = min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] <= max_val_to_survive])

            cases.pop(0)

        selected_individuals.append(random.choice(candidates))

    return selected_individuals


def selBestSum(individuals, k, fit_attr="fitness"):
    return sorted(individuals, key=lambda x: float(np.sum(getattr(x, fit_attr).wvalues)), reverse=True)[:k]


def selMOEAD(individuals, k):
    fun = lambda x: np.sum(softmax(np.random.normal(size=(len(x.fitness.wvalues)))) * np.array(x.fitness.wvalues))
    return sorted(individuals, key=lambda x: float(fun(x)), reverse=True)[:k]


def selIBEA(population, alpha=None, kappa=.05):
    """IBEA Selector"""

    if alpha is None:
        alpha = len(population)

    # Calculate a matrix with the fitness components of every individual
    components = _calc_fitness_components(population, kappa=kappa)

    # Calculate the fitness values
    _calc_fitnesses(population, components)

    # Do the environmental selection
    population[:] = _environmental_selection(population, alpha)

    return population
