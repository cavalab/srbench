"""variation that adds random individual(s)

This module wraps a variation in order to supply random individual(s) to the
offspring after the variation is carried out.
"""
import numpy as np

from .variation import Variation


class AddRandomIndividuals(Variation):
    """A variation object that takes in an implementation of variation that
    adds a random individual to the population before performing variation.

    Parameters
    ----------
    variation : variation
        variation object that performs the variation among individuals
    chromosome_generator : Generator
        Generator for random individual
    num_rand_indvs : int
        The number of random individuals to generate per call
    """
    def __init__(self, variation, chromosome_generator, num_rand_indvs=1):
        super().__init__()
        self._variation = variation
        self._chromosome_generator = chromosome_generator
        self._num_rand_indvs = num_rand_indvs

    def __call__(self, population, number_offspring):
        """Generates a number of random individuals and adds the to the
        population then performs variation on the new population.

        Parameters
        ----------
        population : list of chromosomes
            The population on which to perform variation
        number_offspring : int
            number of offspring to produce.

        Returns
        -------
        list of chromosomes:
            The offspring of the original population and the
            new random individuals
        """
        children = self._variation(population, number_offspring)
        self.mutation_offspring = self._variation.mutation_offspring
        self.crossover_offspring = self._variation.crossover_offspring
        self.offspring_parents = self._variation.offspring_parents
        return self._generate_new_pop(children)

    def _generate_new_pop(self, population):
        for _ in range(self._num_rand_indvs):
            random_indv = self._chromosome_generator()
            population.append(random_indv)
        self.offspring_parents.extend([[]] * self._num_rand_indvs)
        self.crossover_offspring = np.hstack((self.crossover_offspring,
                                              [False] * self._num_rand_indvs))
        self.mutation_offspring = np.hstack((self.mutation_offspring,
                                             [False] * self._num_rand_indvs))
        return population
