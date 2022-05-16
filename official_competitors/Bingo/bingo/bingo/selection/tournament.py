"""Tournament selection

Tournament selection involves running several "tournaments" among a few
individuals (or "chromosomes") chosen at random from the population. The winner
of each tournament (the one with the smallest fitness) is selected to advance
into the next generation.
"""
from operator import attrgetter
import numpy as np

from .selection import Selection
from ..util.argument_validation import argument_validation


class Tournament(Selection):
    """Tournament selection

    This class defines the function for tournament selection in a population.
    In the tournaments random indivduals from the population are chosen; the
    most fit individual from that set advances to the next generation.
    Tournaments repeat until the target population size for the next generation
    is met.

    Parameters
    ----------
    tournament_size : int
        The size of the tournaments
    """
    @argument_validation(tournament_size={">=": 1})
    def __init__(self, tournament_size):
        self._size = tournament_size

    def __call__(self, population, target_population_size):
        """Performs Tournament selection on individuals.

        Parameters
        ----------
        population : list of chromosomes
            The population on which to perform selection
        target_population_size : int
            Target size of the population after selection

        Returns
        -------
        list of chromosomes :
            A subset of the input population (repeats possible)
        """
        next_generation = []
        for _ in range(target_population_size):
            tournament_members = np.random.choice(population, self._size,
                                                  replace=False)
            winner = min(tournament_members, key=attrgetter('fitness'))
            next_generation.append(winner.copy())

        return next_generation
