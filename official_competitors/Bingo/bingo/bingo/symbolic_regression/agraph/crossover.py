"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np

from ...chromosomes.crossover import Crossover


class AGraphCrossover(Crossover):
    """Crossover between acyclic graph individuals
    """

    def __call__(self, parent_1, parent_2):
        """Single point crossover.

        Parameters
        ----------
        parent_1 : `AGraph`
            The first parent individual
        parent_2 : `AGraph`
            The second parent individual

        Returns
        -------
        tuple(`AGraph`, `AGraph`) :
            The two children from the crossover.
        """

        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        ag_size = parent_1.command_array.shape[0]
        cross_point = np.random.randint(1, ag_size-1)
        child_1.mutable_command_array[cross_point:] = \
            parent_2.command_array[cross_point:]
        child_2.mutable_command_array[cross_point:] = \
            parent_1.command_array[cross_point:]

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age

        return child_1, child_2
