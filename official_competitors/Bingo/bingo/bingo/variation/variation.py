"""The genetic operation of variation.

This module defines the basis of variation in bingo evolutionary analyses.
Generally this consists of crossover, mutation and replication.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Variation(metaclass=ABCMeta):
    """A variator of individuals.

    An abstract base class for the variation of genetic populations
    (list chromosomes) in bingo.

    Attributes
    ----------
    crossover_offspring : list of bool
        list indicating whether the corresponding member of the last offspring
        was a result of crossover
    mutation_offspring : list of bool
        list indicating whether the corresponding member of the last offspring
        was a result of mutation
    offspring_parents : list of list of int
        list indicating the parents (by index in the population) of the
        corresponding member of the last offspring
    """
    def __init__(self):
        self.crossover_offspring = np.zeros(shape=(0, ), dtype=bool)
        self.mutation_offspring = np.zeros(shape=(0, ), dtype=bool)
        self.offspring_parents = []

    @abstractmethod
    def __call__(self, population, number_offspring):
        """Performs variation on a population.

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
        raise NotImplementedError
