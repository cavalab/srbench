"""The "Mu + Lambda"

This module defines the basis of the "mu plus lambda" evolutionary algorithm in
bingo analyses. The next generation is evaluated and selected from both the
parent and offspring populations.
"""
from .evolutionary_algorithm import EvolutionaryAlgorithm
from ..variation.var_or import VarOr


class MuPlusLambda(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the "mu plus lambda" evolutionary algorithm in bingo.

    Parameters
    ----------
    evaluation : evaluation
        The evaluation algorithm that sets the fitness on the population.
    selection : selection
        selection instance to perform selection on a population
    crossover : Crossover
        The algorithm that performs crossover during variation.
    mutation : Mutation
        The algorithm that performs mutation during variation.
    crossover_probability : float
        Probability that crossover will occur on an individual.
    mutation_probability : float
        Probability that mutation will occur on an individual.
    number_offspring : int
        The number of offspring produced from variation.
    target_population_size : int (optional)
        The targeted population size. Default is to keep the population the
        same size as the starting population

    Attributes
    ----------
    variation : VarOr
        VarOr variation to perform variation on a population
    evaluation : evaluation
        evaluation instance to perform evaluation on a population
    selection : selection
        selection instance to perform selection on a population
    diagnostics : `bingo.evolutionary_algorithms.ea_diagnostics.EaDiagnostics`
        Public to the EA diagnostics
    """
    def __init__(self, evaluation, selection, crossover, mutation,
                 crossover_probability, mutation_probability,
                 number_offspring, target_population_size=None):
        super().__init__(variation=VarOr(crossover, mutation,
                                         crossover_probability,
                                         mutation_probability),
                         evaluation=evaluation,
                         selection=selection)
        self._number_offspring = number_offspring
        self._target_populations_size = target_population_size

    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of chromosomes
                     The population at the start of the generational step

        Returns
        -------
        list of chromosomes :
            The next generation of the population
        """
        offspring = self.variation(population, self._number_offspring)
        self.evaluation(population)
        self.evaluation(offspring)
        if self._target_populations_size is None:
            new_pop_size = len(population)
        else:
            new_pop_size = self._target_populations_size
        self.update_diagnostics(population, offspring)
        return self.selection(population + offspring, new_pop_size)
