"""Generation of genetic individuals

This module defines the basis of the generation of individuals in bingo
evolutionary analyses.
"""
from abc import ABCMeta, abstractmethod


class Generator(metaclass=ABCMeta):
    """A generator of individuals.

    An abstract base class for the generation of genetic individuals in
    bingo.
    """
    @abstractmethod
    def __call__(self):
        """Generates individuals

        Returns
        -------
        GeneticIndividual :
            A newly generated individual
        """
        raise NotImplementedError
