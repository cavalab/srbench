"""The genetic operation of evaluation.

This module defines the a basic form of the evaluation phase of bingo
evolutionary algorithms.
"""
from multiprocessing import Pool

class Evaluation:
    """Base phase for calculating fitness of a population.

    A base class for the fitness evaluation of populations of genetic
    individuals (list of chromosomes) in bingo.  All individuals in the
    population are evaluated with a fitness function unless their fitness has
    already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    redundant : bool
        Whether to re-evaluate individuals that have been evaluated previously.
        Default False.
    multiprocess : int or bool
        Number of processes to use in parallel evaluation
        or False for serial evaluation.
        Default False.

    Attributes
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    eval_count : int
        the number of fitness function evaluations that have occurred
    """
    def __init__(self, fitness_function, redundant=False, multiprocess=False):
        self.fitness_function = fitness_function
        self._redundant = redundant
        self._multiprocess = multiprocess

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of chromosomes
                     population for which fitness should be calculated
        """
        if self._multiprocess:
            self._multiprocess_eval(population)
        else:
            self._serial_eval(population)

    def _serial_eval(self, population):
        for indv in population:
            if self._redundant or not indv.fit_set:
                indv.fitness = self.fitness_function(indv)

    def _multiprocess_eval(self, population):
        num_procs = self._multiprocess if isinstance(self._multiprocess, int) \
            else None

        with Pool(processes=num_procs) as pool:
            results = []
            for i, indv in enumerate(population):
                if self._redundant or not indv.fit_set:
                    results.append(
                            pool.apply_async(_fitness_job,
                                             (indv, self.fitness_function, i)))

            for res in results:
                indv, i = res.get()
                population[i] = indv


def _fitness_job(individual, fitness_function, population_index):
    individual.fitness = fitness_function(individual)
    return individual, population_index
