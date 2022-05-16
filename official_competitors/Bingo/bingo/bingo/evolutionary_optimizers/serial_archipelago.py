"""The serial implementation of the Archipelago

This module defines the Archipelago data structure that runs serially on
one processor.
"""
import copy
import logging
import numpy as np

from .archipelago import Archipelago
from ..util.log import DETAILED_INFO

LOGGER = logging.getLogger(__name__)


class SerialArchipelago(Archipelago):
    """An collection of islands that evolve serially.

    Evolution of the Archipelago involves independent evolution of Islands
    combined with periodic migration of individuals between random pairs of
    islands. The evolution occurs on one Island at a time.

    Parameters
    ----------
    template_island : `Island`
        The island that acts as a template for all islands in the archipelago
    num_islands : int
        The number of islands to create in the archipelago's list of islands

    Attributes
    ----------
    islands : list of `Island`
        direct access to the islands in the archipelago
    generational_age: int
        The number of generations the archipelago has been evolved
    hall_of_fame: HallOfFame
        An object containing the best individuals seen in the archipelago
    """
    def __init__(self, template_island, num_islands=2, hall_of_fame=None):
        super().__init__(num_islands, hall_of_fame)
        self._template_island = template_island
        self.islands = self._generate_islands(template_island, num_islands)
        for i in self.islands:
            if i.hall_of_fame is None:
                i.hall_of_fame = copy.deepcopy(self.hall_of_fame)

    def _step_through_generations(self, num_steps):
        for island in self.islands:
            island.evolve(num_steps, hall_of_fame_update=False)

    def _coordinate_migration_between_islands(self):
        LOGGER.log(DETAILED_INFO, "Performing migration between Islands")
        island_partners = self._shuffle_island_indices()

        for i in range(self._num_islands//2):
            self._shuffle_island_and_swap_pairs(island_partners, i)

    def get_best_fitness(self):
        """Gets the fitness of most fit member

        Returns
        -------
         :
            Fitness of best individual in the archipelago
        """
        return self.get_best_individual().fitness

    def get_best_individual(self):
        """Returns the best individual

        Returns
        -------
        `Chromosome` :
            The individual with lowest fitness
        """
        list_of_best_indvs = [i.get_best_individual() for i in self.islands]
        list_of_best_indvs.sort(key=lambda x: x.fitness)
        return list_of_best_indvs[0]

    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        return sum([island.get_fitness_evaluation_count()
                    for island in self.islands])

    def get_ea_diagnostic_info(self):
        """ Gets diagnostic info from the evolutionary algorithm(s)

        Returns
        -------
        `EaDiagnosticsSummary` :
            summary of evolutionary algorithm diagnostics
        """
        all_diagnostics = [i.get_ea_diagnostic_info() for i in self.islands]
        return sum(all_diagnostics)

    @staticmethod
    def _generate_islands(island, num_islands):
        island_list = [copy.deepcopy(island)
                       for _ in range(num_islands)]
        for isl in island_list:
            isl.regenerate_population()
        return island_list

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        np.random.shuffle(indices)
        return indices

    def _shuffle_island_and_swap_pairs(self, island_indexes, pair_number):
        partner_1_index = island_indexes[pair_number * 2]
        partner_2_index = island_indexes[pair_number * 2 + 1]
        LOGGER.debug("    %d <-> %d", partner_1_index, partner_2_index)
        partner_1 = self.islands[partner_1_index]
        partner_2 = self.islands[partner_2_index]
        self._population_exchange_program(partner_1, partner_2)
        partner_1.reset_fitness()
        partner_2.reset_fitness()

    @staticmethod
    def _population_exchange_program(island_1, island_2):
        indvs_to_2 = island_1.dump_fraction_of_population(0.5)
        indvs_to_1 = island_2.dump_fraction_of_population(0.5)
        island_1.population += indvs_to_1
        island_2.population += indvs_to_2

    def _log_evolution(self, start_time):
        pass

    def _get_potential_hof_members(self):
        for island in self.islands:
            island.update_hall_of_fame()
        potential_members = [h for i in self.islands for h in i.hall_of_fame]
        return potential_members
