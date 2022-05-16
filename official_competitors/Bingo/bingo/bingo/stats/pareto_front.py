"""
The Pareto Front is an extension of hall of fame to construct a list of all the
non-dominated individuals.  An individual dominates another if all of it's keys
are not worse and at least one is better (smaller) than the other's keys.
"""
import numpy as np

from .hall_of_fame import HallOfFame


class ParetoFront(HallOfFame):
    """A hall of fame object for storing the pareto front of individuals

    The Pareto front is the group of individuals who are not dominated by any
    other individuals in a population.  Domination is calculated based on two
    keys given to the constructor.

    Parameters
    ----------
    secondary_key : function
        Function used to specify the second key to be used in domination
        calculations in the Pareto front. The signature of the function should
        be `func(chromosome)`
    primary_key : function (optional)
        Function used to specify the second key to be used in domination
        calculations in the Pareto front. Default is the use of an individual's
        fitness attribute. The signature of the function should be
        `func(chromosome)`
    similarity_function : function (optional)
        The function used to identify similar individuals. The signature of the
        function should be`func(chromosome, chromosome)`
    """

    def __init__(self, secondary_key, primary_key=None,
                 similarity_function=None):
        super().__init__(max_size=None,
                         key_function=primary_key,
                         similarity_function=similarity_function)
        self._key_func_2 = secondary_key

    def update(self, population):
        """Update the Pareto front based on the given population

        Parameters
        ----------
        population : list of `Chromosome`
            The list of individuals to be considered for induction into the
            Pareto front
        """
        for indv in population:
            if self._not_dominated(indv) and self._not_similar(indv):
                self._remove_dominated_pf_members(indv)
                self.insert(indv)

    def _not_dominated(self, individual):
        if np.isnan(self._key_func(individual)) or \
                np.isnan(self._key_func_2(individual)):
            return False
        for hof_member in self:
            if self._first_dominates(hof_member, individual):
                return False
        return True

    def _first_dominates(self, first_indv, second_indv):
        first_keys = (self._key_func(first_indv),
                      self._key_func_2(first_indv))
        second_keys = (self._key_func(second_indv),
                       self._key_func_2(second_indv))
        if first_keys[0] > second_keys[0] or first_keys[1] > second_keys[1]:
            return False

        not_equal = first_keys[0] != second_keys[0] or \
            first_keys[1] != second_keys[1]
        return not_equal

    def _remove_dominated_pf_members(self, individual):
        dominated_hof_members = self._get_dominated_hof_members(individual)
        for i in reversed(dominated_hof_members):
            self.remove(i)

    def _get_dominated_hof_members(self, individual):
        dominated_members = []
        for i, hof_member in enumerate(self):
            if self._first_dominates(individual, hof_member):
                dominated_members.append(i)
        return dominated_members

    def __str__(self):
        return '\n'.join(["{}\t{}\t{}".format(key, self._key_func_2(i), i)
                          for key, i in zip(self._keys, self._items)])
