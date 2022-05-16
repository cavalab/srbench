"""Multiple Values for genetic information

This file contains several classes that are used for chromosomes
that contains a list of genetic information.
"""
import numpy as np

from .chromosome import Chromosome
from .mutation import Mutation
from .crossover import Crossover
from .generator import Generator
from ..util.argument_validation import argument_validation


class MultipleValueChromosome(Chromosome):
    """ Multiple value individual

    The constructor for a chromosome that holds a list
    of values as opposed to a single value.

    Parameters
    ----------
    values : list of either ints, floats, or bools
        contains the chromosome's list of values

    Attributes
    ----------
    values : list
             the genetic information of the individual

    """
    def __init__(self, values):
        super().__init__()
        self.values = values

    def __str__(self):
        return str(self.values)

    def distance(self, other):
        """Computes the distance (a measure of similarity) between
        two individuals.

        Parameters
        ----------
        other : MultipleValueChromosome

        Returns
        -------
        dist : float
            The distance between self and another chromosome
        """
        dist = sum([v1 != v2 for v1, v2 in zip(self.values, other.values)])
        return dist


class MultipleValueChromosomeGenerator(Generator):
    """Generation of a population of Multi-Value chromosomes

        Parameters
        ----------
        random_value_function : user defined function
            a function that returns randomly generated values to be used as
            components of the chromosomes.
        values_per_chromosome : int
            the number of values that each chromosome will hold
        """
    @argument_validation(values_per_chromosome={">=": 0})
    def __init__(self, random_value_function, values_per_chromosome):
        super().__init__()
        self._random_value_function = random_value_function
        self._values_per_chromosome = values_per_chromosome

    def __call__(self):
        """Generation of a Multi-Value chromosome

        Returns
        -------
        MultipleValueChromosome :
            A chromosome generated using the random value function
        """
        random_list = self._generate_list(self._values_per_chromosome)
        return MultipleValueChromosome(random_list)

    def _generate_list(self, number_of_values):
        return [self._random_value_function() for _ in range(number_of_values)]


class SinglePointMutation(Mutation):
    """Mutation for multiple valued chromosomes

    Performs single-point mutation on the offspring of a parent chromosome.
    The mutation is performed using a user-defined mutation
    function that must return a single randomly generated value.

    Parameters
    ----------
    mutation_function : user defined function
        a function that returns a random value that will replace (or "mutate")
        a random value in the chromosome.
    """
    def __init__(self, mutation_function):
        super().__init__()
        self._mutation_function = mutation_function

    def __call__(self, parent):
        """Performs single-point mutation using the user-defined
        mutation function passed to the constructor

        Parameters
        ----------
        parent : MultipleValueChromosome
            The parent chromosome that is copied to create
            the child that will undergo mutation

        Returns
        -------
        child : MultipleValueChromosome
                The child chromosome that has undergone mutation
        """
        child = parent.copy()
        child.fit_set = False
        mutation_point = np.random.randint(len(parent.values))
        child.values[mutation_point] = self._mutation_function()
        return child


class SinglePointCrossover(Crossover):
    """Crossover for multiple valued chromosomes

    Crossover results in two individuals with single-point crossed-over lists
    whose values are provided by the parents. Crossover point is
    a random integer produced by numpy.
    """
    def __init__(self):
        super().__init__()
        self._crossover_point = 0

    def __call__(self, parent_1, parent_2):
        """Performs single-point crossover of two parent chromosomes

        Parameters
        ----------
        parent_1 : MultipleValueChromosome
            the first parent to be used for crossover
        parent_2 : MultipleValueChromosome
            the second parent to be used for crossover

        Returns
        -------
        child_1, child_2 : tuple of MultiValueChromosome
            a tuple of the 2 children produced by crossover
        """
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()
        child_1.fit_set = False
        child_2.fit_set = False
        self._crossover_point = np.random.randint(len(parent_1.values))
        child_1.values = parent_1.values[:self._crossover_point] \
                         + parent_2.values[self._crossover_point:]
        child_2.values = parent_2.values[:self._crossover_point] \
                         + parent_1.values[self._crossover_point:]
        if parent_1.genetic_age > parent_2.genetic_age:
            age = parent_1.genetic_age
        else:
            age = parent_2.genetic_age
        child_1.genetic_age = age
        child_2.genetic_age = age
        return child_1, child_2
