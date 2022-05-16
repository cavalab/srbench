"""
This module contains the utilities for fitness predictors.

Fitness predictors in bingo are chromosomes that encode information necessary
to make a prediction of fitness for other chromosomes types.  The type of
fitness predicters used here are subset fitness predictors, which make a
prediction of fitness by using only a subset of the training data used in
preforming a full/true fitness calculation.  Subset fitness predictors use the
MultilpleValueChromosome.

Check out the works of the works of Schmidt and Lipson for more details:
e.g., "Coevolution of Fitness Predictors" (2008) .
"""
import logging
from copy import deepcopy
import numpy as np
from ..util.argument_validation import argument_validation
from ..evaluation.fitness_function import FitnessFunction

LOGGER = logging.getLogger(__name__)


class FitnessPredictorFitnessFunction(FitnessFunction):
    """A fitness function for subset fitness predictors

    A fitness function for fitness predictors. The fitness of a fitness
    predictor is based upon its ability to predict true/full fitness of trainer
    individuals.

    Parameters
    ----------
    training_data : `TrainingData`
        Training data used in full/true fitness evaluation
    full_fitness_function : FitnessFunction
        The fitness function to be used in prediction (based on the training
        data)
    potential_trainers : list of `Chromosome`
        a list of individuals that could potentially be used as trainers
    num_trainers : int
        number of trainers to use
    """
    @argument_validation(num_trainers={">": 0})
    def __init__(self, training_data, full_fitness_function,
                 potential_trainers, num_trainers):
        super().__init__(training_data)
        self._next_trainer_to_update = 0
        self.point_eval_count = 0
        self._fitness_function = deepcopy(full_fitness_function)
        self._trainers, self._true_fitness_for_trainers = \
            self._make_initial_trainer_population(potential_trainers,
                                                  num_trainers)

    def __call__(self, individual):
        """Fitness function for subset fitness predictors

        The average absolute error of the predicted versus true/full fitness of
        the trainers for a given fitness predictor individual.

        Parameters
        ----------
        individual : `MultipleValueChromosome`
            A subset fitness predictor

        Returns
        -------
        float :
            fitness of the predictor
        """
        self.eval_count += 1
        error_in_fitness_predictions = 0.0
        for trainer, true_fitness in zip(self._trainers,
                                         self._true_fitness_for_trainers):
            predicted_fitness = \
                self.predict_fitness_for_trainer(individual, trainer)
            error_in_fitness_predictions += abs(true_fitness
                                                - predicted_fitness)
        return error_in_fitness_predictions / len(self._trainers)

    def add_trainer(self, trainer):
        """Add a trainer to the trainer population

        Replaces one of the current trainers.

        Parameters
        ----------
        trainer : `Chromosome`
            individual to add to the training population
        """
        self._trainers[self._next_trainer_to_update] = trainer.copy()
        self._true_fitness_for_trainers[self._next_trainer_to_update] = \
            self.get_true_fitness_for_trainer(trainer)
        self._increment_next_trainer_to_update()

    def predict_fitness_for_trainer(self, individual, trainer):
        """Get predicted value of fitness for a trainer

        Parameters
        ----------
        individual : `MultipleValueChromosome`
            subset fitness predictor to use in calculating predicted fitness
        trainer : `Chromosome`
            the trainer of which to calculate fitness

        Returns
        -------
        float :
            predicted fitness
        """
        subset_training_data = \
            self.training_data[individual.values]
        self._fitness_function.training_data = subset_training_data
        predicted_fitness = self._fitness_function(trainer)
        self.point_eval_count += len(subset_training_data)
        return predicted_fitness

    def get_true_fitness_for_trainer(self, trainer):
        """Gets true (full) fitness of trainer

        True fitness is the fitness calculated using the entire set of training
        data.

        Parameters
        ----------
        trainer : chromosomes
            The chromosome to be evaluated

        Returns
        -------
         :
            true (full) fitness of trainer
        """
        self._fitness_function.training_data = self.training_data
        predicted_fitness = self._fitness_function(trainer)
        self.point_eval_count += len(self.training_data)
        return predicted_fitness

    def _make_initial_trainer_population(self, potential_trainers,
                                         num_trainers):
        trainers = []
        true_fitness_for_trainers = []
        for indv in potential_trainers:
            true_fitness = self.get_true_fitness_for_trainer(indv)
            if not np.isnan(true_fitness):
                trainers.append(indv.copy())
                true_fitness_for_trainers.append(true_fitness)
            # TODO could implement a check to make sure no fitness predictors
            #  are nan for candidate individual
            if len(trainers) == num_trainers:
                return trainers, true_fitness_for_trainers

        raise RuntimeError("FitnessPredictorFitnessFunction could not be "
                           "initialized. Not enough valid trainers.")

    def _increment_next_trainer_to_update(self):
        self._next_trainer_to_update += 1
        if self._next_trainer_to_update >= len(self._trainers):
            self._next_trainer_to_update -= len(self._trainers)


class FitnessPredictorIndexGenerator:
    """Generator of ints within a range

    Parameters
    ----------
    data_size : int
                maximum value of randomly generated int (non inclusive)
    """
    def __init__(self, data_size):
        self._max = data_size

    def __call__(self):
        return np.random.randint(self._max)
