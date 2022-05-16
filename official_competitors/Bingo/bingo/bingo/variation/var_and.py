"""variation where crossover and mutation may co-occur

var_and.py is similar to the function of the same name in DEAP.  It allows for
definition of a variation by crossover and mutation probabilities. Offspring
may be the result of both crossover :and: mutation, hence the name.
"""
import numpy as np

from .variation import Variation
from ..util.argument_validation import argument_validation


class VarAnd(Variation):
    """variation where crossover and mutation may co-occur

    Parameters
    ----------
    crossover : Crossover
        Crossover function class used in the variation
    mutation : Mutation
        Mutation function class used in the variation
    crossover_probability : float
        Probability that crossover will occur on an individual
    mutation_probability : float
        Probability that mutation will occur on an individual


    Attributes
    ----------
    crossover_offspring : array of bool
        list indicating whether the corresponding member of the last offspring
        was a result of crossover
    mutation_offspring : array of bool
        list indicating whether the corresponding member of the last offspring
        was a result of mutation
    offspring_parents : list of list of int
        list indicating the parents (by index in the population) of the
        corresponding member of the last offspring
    """

    @argument_validation(crossover_probability={">=": 0, "<=": 1},
                         mutation_probability={">=": 0, "<=": 1})
    def __init__(self, crossover, mutation, crossover_probability,
                 mutation_probability):
        super().__init__()
        self._crossover = crossover
        self._mutation = mutation
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability

    @argument_validation(number_offspring={">=": 0})
    def __call__(self, population, number_offspring):
        """Performs "And" variation on a population.

        Parameters
        ----------
        population : list of chromosomes
            The population on which to perform selection
        number_offspring : int
            number of offspring to produce

        Returns
        -------
        list of chromosomes :
            The offspring of the population
        """
        self.crossover_offspring = np.zeros(number_offspring, bool)
        self.mutation_offspring = np.zeros(number_offspring, bool)
        self.offspring_parents = [[]] * number_offspring
        offspring = self._crossover_population(number_offspring, population)
        self._mutate_population(offspring)
        return offspring

    def _crossover_population(self, number_offspring, population):
        offspring = []
        for i in range(0, number_offspring - 1, 2):
            parent_index_1 = i % len(population)
            parent_index_2 = (parent_index_1 + 1) % len(population)
            if np.random.random() <= self._crossover_probability:
                child_1, child_2 = self._crossover(population[parent_index_1],
                                                   population[parent_index_2])
                offspring.append(child_1)
                offspring.append(child_2)
                self.crossover_offspring[i:i + 2] = True
                self.offspring_parents[i] = [parent_index_1, parent_index_2]
                self.offspring_parents[i + 1] = \
                    [parent_index_1, parent_index_2]
            else:
                offspring.append(population[parent_index_1].copy())
                offspring.append(population[parent_index_2].copy())
                self.offspring_parents[i] = [parent_index_1]
                self.offspring_parents[i + 1] = [parent_index_2]
        if len(offspring) < number_offspring:
            parent_index_1 = (len(offspring) + 1) % len(population)
            offspring.append(population[parent_index_1].copy())
            self.offspring_parents[-1] = [parent_index_1]
        return offspring

    def _mutate_population(self, offspring):
        for i, parent in enumerate(offspring):
            if np.random.random() <= self._mutation_probability:
                offspring[i] = self._mutation(parent)
                self.mutation_offspring[i] = True
