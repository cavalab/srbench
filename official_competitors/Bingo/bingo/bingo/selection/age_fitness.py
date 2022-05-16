"""Age-Fitness selection

This module implements the Age-Fitness selection algorithm that defines
the selection used in the Age-Fitness evolutionary algorithm module.
This module expects to be used in conjunction with the
``RandomIndividualVariation`` module that wraps the ``VarOr`` module.
"""
import numpy as np

from .selection import Selection
from ..util.argument_validation import argument_validation


class AgeFitness(Selection):
    """Age-Fitness selection

    Parameters
    ----------
    selection_size : int
        The size of the group of individuals to be randomly compared. The size
        must be an integer greater than 1.
    """
    WORST_CASE_FACTOR = 50

    @argument_validation(selection_size={">=": 2})
    def __init__(self, selection_size=2):
        self._selection_size = selection_size
        self._selected_indices = []
        self._population_index_array = np.array([])
        self._selection_attempts = 0

    def __call__(self, population, target_population_size):
        """Performs Age-Fitness selection on a population. If ``selection_size``
        is larger than the population, the population size is used as the
        ``selection_size``.

        Parameters
        ----------
        population : list of chromosomes
            The population on which to perform selection
        target_population_size : int
            The size of the new population after selection. It will never be
            the case that the new population will have a size smaller than the
            target population. However, it *is* possible to for the new
            population to be larger than ``target_population_size``.

        Returns
        -------
        list of chromosomes :
            The chromosomes not selected for removal

        Raises
        ------
        ValueError
            If the ``target_population_size`` is larger than the intial
            `population`
        """
        if target_population_size > len(population):
            raise ValueError("Target population size should\
                              be less than initial population")

        n_removed = 0
        start_pop_size = len(population)
        target_removal = start_pop_size - target_population_size

        selection_attempts = 0
        while n_removed < target_removal and \
                selection_attempts < start_pop_size * self.WORST_CASE_FACTOR:

            inds = self._get_unique_rand_indices(start_pop_size - n_removed)
            to_remove = self._find_inds_for_removal(inds, population,
                                                    target_removal - n_removed)
            self._swap_removals_to_end(population, to_remove, n_removed)

            n_removed += len(to_remove)
            selection_attempts += 1

        new_pop_size = start_pop_size - n_removed
        return population[:new_pop_size]

    def _get_unique_rand_indices(self, max_int):
        if self._selection_size >= max_int:
            return list(range(max_int))
        if self._selection_size < 5:
            return self._dumb_selection(max_int)
        return np.random.choice(max_int, self._selection_size, replace=False)

    def _dumb_selection(self, max_int):
        inds = set(np.random.randint(max_int, size=self._selection_size))
        while len(inds) < self._selection_size:
            inds.add(np.random.randint(max_int))
        return list(inds)

    def _find_inds_for_removal(self, inds, population, num_removals_needed):
        if self._selection_size == 2:
            return self._streamlined_pair_removal(inds[0], inds[1], population)

        removal_set = set()
        for i, ind_a in enumerate(inds[:-1]):
            if ind_a not in removal_set:
                for ind_b in inds[i+1:]:
                    if ind_b not in removal_set:
                        self._update_removal_set(population, ind_a, ind_b,
                                                 removal_set)
                        if len(removal_set) >= num_removals_needed:
                            return list(removal_set)
        return list(removal_set)

    def _streamlined_pair_removal(self, indv_index_1, indv_index_2,
                                  population):
        indv_1 = population[indv_index_1]
        indv_2 = population[indv_index_2]

        if np.isnan(indv_1.fitness):
            return [indv_index_1]
        if np.isnan(indv_2.fitness):
            return [indv_index_2]
        if self._first_not_dominated(indv_1, indv_2):
            return [indv_index_2]
        if self._first_not_dominated(indv_2, indv_1):
            return [indv_index_1]
        return []

    def _update_removal_set(self, population, indv_index_1,
                            indv_index_2, removal_set):
        indv_1 = population[indv_index_1]
        indv_2 = population[indv_index_2]

        if np.isnan(indv_1.fitness):
            removal_set.add(indv_index_1)
        elif np.isnan(indv_2.fitness):
            removal_set.add(indv_index_2)
        elif self._first_not_dominated(indv_1, indv_2):
            removal_set.add(indv_index_2)
        elif self._first_not_dominated(indv_2, indv_1):
            removal_set.add(indv_index_1)

    @staticmethod
    def _first_not_dominated(first_indv, second_indv):
        # This code block can be used to force equivalency of bingocpp and
        # bingo that may otherwise diverge because of truncation or small math
        # differences.
        # rel_fitness_diff = (first_indv.fitness - second_indv.fitness) \
        #                   / (first_indv.fitness + second_indv.fitness)
        # return not (first_indv.genetic_age > second_indv.genetic_age or
        #             rel_fitness_diff > 1e-15)
        return not (first_indv.genetic_age > second_indv.genetic_age or
                    first_indv.fitness > second_indv.fitness)

    def _swap_removals_to_end(self, population, inds_to_remove, num_removed):
        for i, ind in enumerate(sorted(inds_to_remove, reverse=True)):
            self._swap(population, ind, -(i+num_removed+1))

    @staticmethod
    def _swap(array, index_1, index_2):
        array[index_1], array[index_2] = array[index_2], array[index_1]
