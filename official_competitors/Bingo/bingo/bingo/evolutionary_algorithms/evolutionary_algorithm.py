"""The base of evolutionary algorithm definition

This module defines the basis of evolutionary algorithms in bingo analyses.
An evolutionary algorithms in bingo is defined by three phases: variation,
evaluation, and selection.  These phases, when repeated, define the evolution
of a population.
"""
from .ea_diagnostics import EaDiagnostics


class EvolutionaryAlgorithm:
    """The algorithm used to perform generational steps.

    The basic implementation used in this base Base implementation is a simple
    steady-state Base (akin to simpleEA in DEAP)

    Parameters
    ----------
    variation : `variation`
        The phase of bingo EAs that is responsible for varying the population
        (usually through some form of crossover and/or mutation).
    evaluation : `evaluation`
        The phase in bingo EAs responsible for the evaluation of the fitness of
        the individuals in the population.
    selection : `selection`
        The phase of bingo EAs responsible for selecting the individuals in a
        population which survive into the next generation.

    Attributes
    ----------
    variation : `variation`
        Public access to the variation phase of the EA
    evaluation : `evaluation`
        Public access to the evaluation phase of the EA
    selection : `selection`
        Public access to the selection phase of the EA
    diagnostics : `bingo.evolutionary_algorithms.ea_diagnostics.EaDiagnostics`
        Public to the EA diagnostics
    """
    def __init__(self, variation, evaluation, selection):
        self.variation = variation
        self.evaluation = evaluation
        self.selection = selection
        self.diagnostics = EaDiagnostics()

    def generational_step(self, population):
        """Performs a generational step on population.

        Parameters
        ----------
        population : list of chromosomes
            The population at the start of the generational step

        Returns
        -------
        list of chromosomes :
            The next generation of the population
        """
        population_size = len(population)
        offspring = self.variation(population, population_size)
        self.evaluation(population)  # TODO (David Randall) why do we split up the evaluation
                                     #   it doesn't seem to apply more parallelization imo
        self.evaluation(offspring)
        next_generation = self.selection(offspring, population_size)
        self.update_diagnostics(population, offspring)
        return next_generation

    def update_diagnostics(self, population, offspring):
        """
        Update the evolutionary algorithms diagnostic information based on a
        new generation of offspring

        Parameters
        ----------
        population: list of `Chromosome`
            The original population fo the generation
        offspring: list of `Chromosome`
            The potential new members of the population

        """
        self.diagnostics.update(population, offspring,
                                self.variation.offspring_parents,
                                self.variation.crossover_offspring,
                                self.variation.mutation_offspring)
