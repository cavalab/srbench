"""An Island implementation using fitness predictors

The primary goal of this island is to be more computationally efficient than
the simple Island in cases where lots of TrainingData is needed for fitness
evaluation.  Fitness predictors use a subset of that training data to make a
prediction of what fitness would be using the whole data set.

The fitness predictors are evolved concurrently with the main island population
so that their predictive power is growing as the population becomes more fit.
A full description of the method can be found in the works of Schmidt and
Lipson [1]_.

.. [1] Schmidt, Michael D., and Hod Lipson. "Coevolution of fitness
       predictors." IEEE Transactions on Evolutionary Computation 12.6 (2008):
       736-749.
"""
import logging
from copy import copy, deepcopy
import numpy as np
from ..util.argument_validation import argument_validation
from ..evaluation.evaluation import Evaluation
from ..evolutionary_algorithms.deterministic_crowding \
    import DeterministicCrowdingEA
from .island import Island
from ..chromosomes.multiple_values import MultipleValueChromosomeGenerator, \
                                         SinglePointCrossover, \
                                         SinglePointMutation
from .fitness_predictor import FitnessPredictorFitnessFunction, \
                              FitnessPredictorIndexGenerator

LOGGER = logging.getLogger(__name__)


class FitnessPredictorIsland(Island):
    """An island utilizing co-evolving fitness predictors

    The coevolution of fitness predictors is a evolutionary scheme meant to
    reduce computation cost and reduce overfitting and bloat.  It is similar
    in spirit to the use of mini-batches in stochastic gradient descent.

    Parameters
    ----------
    evolution_algorithm : EvolutionaryAlgorithm
        The desired algorithm to use in assessing the population
    generator : Generator
        The generator class that returns an instance of a chromosome
    population_size : int
        The desired size of the population
    predictor_population_size  : int (Optional)
        The number of co-evolving fitness predictors. Default is 16.
    predictor_update_frequency  : int (Optional)
        The number of generations between updates to the main fitness function.
        Default is 50.
    predictor_size_ratio : float (0 - 1] (Optional)
        The fraction of the training data  that will be used by fitness
        predictors.  Default is 0.1.
    predictor_computation_ratio : float [0 - 1) (Optional)
        The fraction of total computation that is devoted to evolution of
        fitness predictors. Default is 0.1.
    trainer_population_size  : int (Optional)
        The number of individuals used in training of fitness predictors.
        Default is 16.
    trainer_update_frequency  : int (Optional)
        The number of generations between updates to the trainer population.
        Default is 50.
    hall_of_fame : HallOfFame (Optional)
        The hall of fame object to be used for storing best individuals

    Attributes
    ----------
    generational_age : int
        The number of generational steps that have beenexecuted
    population : list of chromosomes
        The population that is evolving
    hall_of_fame: HallOfFame
        An object containing the best individuals seen in the optimization
    """
    @argument_validation(population_size={">=": 0},
                         predictor_population_size={">=": 0},
                         predictor_update_frequency={">": 0},
                         predictor_size_ratio={">": 0, "<=": 1},
                         predictor_computation_ratio={">=": 0, "<": 1},
                         trainer_population_size={">=": 0},
                         trainer_update_frequency={">": 0})
    def __init__(self, evolution_algorithm, generator, population_size,
                 predictor_population_size=16, predictor_update_frequency=50,
                 predictor_size_ratio=0.1, predictor_computation_ratio=0.1,
                 trainer_population_size=16, trainer_update_frequency=50,
                 hall_of_fame=None):
        super().__init__(evolution_algorithm, generator, population_size,
                         None)

        self._hof_w_true_fitness = hall_of_fame
        self._hof_w_predicted_fitness = deepcopy(hall_of_fame)
        self._potential_hof_members = {}

        self._fitness_function = self._ea.evaluation.fitness_function
        self._full_training_data = copy(self._fitness_function.training_data)
        self._full_data_size = len(self._full_training_data)

        self._predictor_population_size = predictor_population_size
        self._predictor_size = max((
            int(predictor_size_ratio * self._full_data_size),
            min((10, self._full_data_size))
        ))
        self._predictor_update_frequency = predictor_update_frequency
        # pylint: disable=C0103
        self._target_predictor_computation_ratio = predictor_computation_ratio

        self._trainer_population_size = trainer_population_size
        self._trainer_update_frequency = trainer_update_frequency

        self._predictor_fitness_function = \
            self._make_fitness_predictor_fitness_function()

        self._predictor_island = self._make_predictor_island()
        self._update_to_use_best_fitness_predictor()

    @property
    def hall_of_fame(self):
        """The hall of fame object which is updated during evolution"""
        return self._hof_w_true_fitness

    @hall_of_fame.setter
    def hall_of_fame(self, hall_of_fame):
        self._hof_w_true_fitness = hall_of_fame
        self._hof_w_predicted_fitness = deepcopy(hall_of_fame)

    def _execute_generational_step(self):
        LOGGER.debug("I> %d", self.generational_age + 1)
        super()._execute_generational_step()

        self._step_predictor_island_to_maintain_ratio()
        self._update_predictor_if_needed()
        self._update_trainer_if_needed()

    def _make_fitness_predictor_fitness_function(self):
        pred_fit_func = \
            FitnessPredictorFitnessFunction(self._full_training_data,
                                            self._fitness_function,
                                            self.population,
                                            self._trainer_population_size)
        return pred_fit_func

    def _make_predictor_island(self):
        index_generator = FitnessPredictorIndexGenerator(self._full_data_size)
        predictor_ea = self._make_predictor_ea(index_generator)
        generator = MultipleValueChromosomeGenerator(index_generator,
                                                     self._predictor_size)
        predictor_island = Island(predictor_ea, generator,
                                  self._predictor_population_size)
        predictor_island.evolve(1, suppress_logging=True)
        return predictor_island

    def _make_predictor_ea(self, index_generator):
        crossover = SinglePointCrossover()
        mutation = SinglePointMutation(index_generator)
        evaluation = Evaluation(self._predictor_fitness_function)
        dc_ea = DeterministicCrowdingEA(evaluation, crossover, mutation,
                                        crossover_probability=0.5,
                                        mutation_probability=0.2)
        return dc_ea

    def _step_predictor_island_to_maintain_ratio(self):
        while (self._get_predictor_computation_ratio()
               < self._target_predictor_computation_ratio):
            LOGGER.debug("P> %d", self._predictor_island.generational_age + 1)
            self._predictor_island.evolve(1, suppress_logging=True)

    def _update_predictor_if_needed(self):
        if self.generational_age % self._predictor_update_frequency == 0:
            LOGGER.debug("Updating fitness predictor")
            self._update_to_use_best_fitness_predictor()
            self.reset_fitness(self.population)
            if self._hof_w_predicted_fitness is not None:
                self._hof_w_predicted_fitness.clear()
            self.evaluate_population()

    def _update_trainer_if_needed(self):
        if self.generational_age % self._trainer_update_frequency == 0:
            LOGGER.debug("Updating trainer")
            self._add_new_trainer()
            self.reset_fitness(self._predictor_island.population)
            self._predictor_island.evaluate_population()

    def _update_to_use_best_fitness_predictor(self):
        best_predictor = self._predictor_island.get_best_individual()
        best_subset_data = \
            self._full_training_data[best_predictor.values]
        self._fitness_function.training_data = best_subset_data

    def _add_new_trainer(self):
        best_candidate = self._find_best_new_trainer()
        self._predictor_fitness_function.add_trainer(best_candidate.copy())

    def _find_best_new_trainer(self):
        best_candidate = self.population[0]
        max_variance = 0
        for individual in self.population:
            variance = self._calculate_predictor_variance_of(individual)
            if variance > max_variance:
                max_variance = variance
                best_candidate = individual
        return best_candidate

    def _calculate_predictor_variance_of(self, individual):
        predicted_fitness_list = \
            [self._predictor_fitness_function.predict_fitness_for_trainer(
                predictor, individual)
             for predictor in self._predictor_island.population]
        try:
            variance = np.var(predicted_fitness_list)
        except (ArithmeticError, OverflowError, FloatingPointError,
                ValueError):
            variance = np.nan
        return variance

    def _get_predictor_computation_ratio(self):
        predictor_expense = self._predictor_fitness_function.point_eval_count
        island_expense = self._fitness_function.eval_count \
                         * self._predictor_size
        return predictor_expense / (predictor_expense + island_expense)

    def _get_potential_hof_members(self):
        self._hof_w_predicted_fitness.update(self.population)
        potential_members = []
        for indv_w_ped_fitness in self._hof_w_predicted_fitness:
            indv_w_true_fitness = deepcopy(indv_w_ped_fitness)
            indv_w_true_fitness.fitness = \
                self._predictor_fitness_function.get_true_fitness_for_trainer(
                    indv_w_true_fitness)
            potential_members.append(indv_w_true_fitness)
        return potential_members

    def get_best_individual(self):
        """Finds the individual with the lowest fitness in a population.

        This assures  that the fitness is the true ftness value and not the
        predicted fitness.

        Returns
        -------
        best : chromosomes
            The chromosomes with the lowest fitness value
        """
        best_indv = super().get_best_individual().copy()
        best_indv.fitness = \
            self._predictor_fitness_function.get_true_fitness_for_trainer(
                best_indv)
        return best_indv
