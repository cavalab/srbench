""" The abstract-base class for the Archipelago.

This module defines the data structure that manages a group of `Islands`.
Archipelago implementations will control the generational steps for all
the islands until convergence or until a maximal number of steps is reached.
"""
from abc import ABCMeta, abstractmethod

from .evolutionary_optimizer import EvolutionaryOptimizer


class Archipelago(EvolutionaryOptimizer, metaclass=ABCMeta):
    """A collection of islands

    Evolution of the Archipelago involves independent evolution of Islands
    combined with periodic migration of individuals between random pairs of
    islands.

    Parameters
    ----------
    template_island : Island
        Island that will be used as a template for islands in the archipelago
    num_islands : int
        The size of the archipelago; the number of islands it contains

    Attributes
    ----------
    generational_age: int
        The number of generations the archipelago has been evolved
    hall_of_fame: HallOfFame
        An object containing the best individuals seen in the archipelago
    """
    def __init__(self, num_islands, hall_of_fame=None):
        super().__init__(hall_of_fame)
        self._num_islands = num_islands

    def _do_evolution(self, num_generations):
        self._coordinate_migration_between_islands()
        self._step_through_generations(num_generations)
        self.generational_age += num_generations

    @abstractmethod
    def _step_through_generations(self, num_steps):
        raise NotImplementedError

    @abstractmethod
    def _coordinate_migration_between_islands(self):
        raise NotImplementedError
