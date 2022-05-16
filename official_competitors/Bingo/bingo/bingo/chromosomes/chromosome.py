"""chromosomes: the base class for all genetic individuals in bingo

This module defines a chromosomes, the base class for all genetic individuals in
bingo evolutionary analyses.  Extension of bingo can be performed by developing
subclasses of chromosomes.
"""

import copy
from abc import ABCMeta, abstractmethod


class Chromosome(metaclass=ABCMeta):
    """A genetic individual

    This class is the base of a genetic individual in bingo evolutionary
    analyses.

    Parameters
    ----------
    genetic_age : int
        age of the oldest component of the genetic material in the individual
    fitness :
        starting value of fitness
    fit_set : bool
        Whether the fitness has been calculated for the individual

    Attributes
    ----------
    fitness
    genetic_age : int
                  age of the oldest component of the genetic material in the
                  individual
    fit_set : bool
              Whether the fitness has been calculated for the individual
    """
    def __init__(self, genetic_age=0, fitness=None, fit_set=False):
        self._genetic_age = genetic_age
        self._fitness = fitness
        self._fit_set = fit_set

    @property
    def fitness(self):
        """numeric or tuple of numeric: The fitness of the individual"""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness
        self._fit_set = True

    @property
    def genetic_age(self):
        """The age of the oldest components of the individual"""
        return self._genetic_age

    @genetic_age.setter
    def genetic_age(self, genetic_age):
        self._genetic_age = genetic_age

    @property
    def fit_set(self):
        """Indication of whether the fitness has been set"""
        return self._fit_set

    @fit_set.setter
    def fit_set(self, fit_set):
        self._fit_set = fit_set

    def copy(self):
        """copy

        Returns
        -------
            A deep copy of self
        """
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        """String conversion of individual

        Returns
        -------
        str
            individual string form
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, chromosome):
        """Distance from self to chromosome

        Returns
        -------
        float
            distance from self to chromosome
        """
        raise NotImplementedError
