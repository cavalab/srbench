"""The definition of fitness evalutions for individuals.

This module defines the basis of fitness evaluation in bingo evolutionary
analyses.
"""
from abc import ABCMeta, abstractmethod

import numpy as np


# Fitness metric functions, outside of FitnessFunction for use in GradientMixin
def mean_absolute_error(vector):
    """Calculate the mean absolute error of an error vector"""
    return np.mean(np.abs(vector))


def root_mean_squared_error(vector):
    """Calculate the root mean squared error of an error vector"""
    return np.sqrt(np.mean(np.square(vector)))


def mean_squared_error(vector):
    """Calculate the mean squared error of an error vector"""
    return np.mean(np.square(vector))


class FitnessFunction(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (chromosomes) in bingo.

    Parameters
    ----------
    training_data :
        Optional) data that can be used in fitness evaluation

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed
    training_data :
        (Optional) data that can be used in fitness evaluation
    """
    def __init__(self, training_data=None):
        self.eval_count = 0
        self.training_data = training_data

    @abstractmethod
    def __call__(self, individual):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        individual : chromosomes
            individual for which fitness will be calculated

        Notes
        -----
        The eval_count should be incremented in a subclass' __call__ definition
        for accurate evaluation counting

        Returns
        -------
         :
            fitness of the individual
        """
        raise NotImplementedError


class VectorBasedFunction(FitnessFunction, metaclass=ABCMeta):
    """Fitness evaluation based on vectorized fitness

    Parameters
    ----------
    training_data : ExplicitTrainingData
        data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', and
        'root mean squared error'
    """
    def __init__(self, training_data=None, metric="mae"):
        super().__init__(training_data)

        if metric in ["mean absolute error", "mae"]:
            self._metric = mean_absolute_error
        elif metric in ["mean squared error", "mse"]:
            self._metric = mean_squared_error
        elif metric in ["root mean squared error", "rmse"]:
            self._metric = root_mean_squared_error
        else:
            raise ValueError("Invalid metric for Fitness Function")

    def __call__(self, individual):
        """Vector based fitness evaluation

        Evaluate the fitness of an individual as based on a vector of fitness
        (error) values.  The metric defined in the consructor is used to
        aggregate the vector fitness into a single fitness value

        Parameters
        ----------
        individual : chromosomes
            individual for which fitness will be calculated

        Returns
        -------
         :
           fitness of the individual
        """
        fitness_vector = self.evaluate_fitness_vector(individual)
        return self._metric(fitness_vector)

    @abstractmethod
    def evaluate_fitness_vector(self, individual):
        """Calaculate a vector of fitness values"""
        raise NotImplementedError
