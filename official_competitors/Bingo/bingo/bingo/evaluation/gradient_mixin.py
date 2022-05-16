"""Mixin classes used to extend fitness functions to be able to use
gradient- and jacobian-based continuous local optimization methods.

This module defines the basis of gradient and jacobian partial derivatives
of fitness functions used in bingo evolutionary analyses.
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from .fitness_function \
    import mean_absolute_error, mean_squared_error, root_mean_squared_error


class GradientMixin(metaclass=ABCMeta):
    """Mixin for using gradients for fitness functions

    An abstract base class/mixin used to implement the gradients
    of fitness functions.
    """
    @abstractmethod
    def get_fitness_and_gradient(self, individual):
        """Fitness function evaluation and gradient

        Get the fitness of the individual and the gradient
        of this function with respect to the individual's constants.

        Parameters
        ----------
        individual : chromosomes
            individual for which the fitness and gradient will be calculated for

        Returns
        -------
        fitness, gradient :
            fitness of the individual and the gradient of this function
            with respect to the individual's constants
        """
        raise NotImplementedError


class VectorGradientMixin(GradientMixin):
    """Mixin for using gradients and jacobians in vector based fitness functions

    An abstract base class/mixin used to implement the gradients and jacobians
    of vector based fitness functions.

    Parameters
    ----------
    training_data : ExplicitTrainingData
        data that is used in fitness evaluation (passed to parent).
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', and
        'root mean squared error'
    """
    def __init__(self, training_data=None, metric="mae"):
        super().__init__(training_data, metric)

        if metric in ["mean absolute error", "mae"]:
            self._metric = mean_absolute_error
            self._metric_derivative = \
                VectorGradientMixin._mean_absolute_error_derivative
        elif metric in ["mean squared error", "mse"]:
            self._metric = mean_squared_error
            self._metric_derivative = \
                VectorGradientMixin._mean_squared_error_derivative
        elif metric in ["root mean squared error", "rmse"]:
            self._metric = root_mean_squared_error
            self._metric_derivative = \
                VectorGradientMixin._root_mean_squared_error_derivative
        else:
            raise ValueError("Invalid metric for vector gradient mixin")

    def get_fitness_and_gradient(self, individual):
        """Fitness evaluation and gradient of vector based fitness
        function using metric (i.e. the fitness function returns
        a vector that is converted into a scalar using its metric function)

        Get the fitness of the individual and the gradient
        of this function with respect to the individual's constants.

        Parameters
        ----------
        individual : chromosomes
            individual for which the fitness and gradient will be calculated for

        Returns
        -------
        fitness, gradient :
            fitness of the individual and the gradient of this function
            with respect to the individual's constants
        """
        fitness_vector, jacobian = \
            self.get_fitness_vector_and_jacobian(individual)
        return self._metric(fitness_vector), \
               self._metric_derivative(fitness_vector, jacobian.transpose())

    @abstractmethod
    def get_fitness_vector_and_jacobian(self, individual):
        """Returns the vectorized fitness of this individual and
        the jacobian of this vector fitness function with
        respect to the individual's constants

        jacobian = [[:math:`df1/dc1`, :math:`df1/dc2`, ...],
                    [:math:`df2/dc1`, :math:`df2/dc2`, ...],
                    ...]
            where :math:`f` # is the fitness function corresponding with the
            #th fitness vector entry and :math:`c` # is the corresponding
            constant of the individual

        Parameters
        ----------
        individual : chromosomes
            individual used for vectorized fitness evaluation and jacobian
            calculation

        Returns
        -------
        fitness_vector, jacobian :
            the vectorized fitness of the individual and
            the partial derivatives of each fitness function with respect
            to the individual's constants
        """
        raise NotImplementedError

    @staticmethod
    def _mean_absolute_error_derivative(fitness_vector, fitness_partials):
        return np.mean(np.sign(fitness_vector) * fitness_partials, axis=1)

    @staticmethod
    def _mean_squared_error_derivative(fitness_vector, fitness_partials):
        return 2 * np.mean(fitness_vector * fitness_partials, axis=1)

    @staticmethod
    def _root_mean_squared_error_derivative(fitness_vector, fitness_partials):
        return 1/np.sqrt(np.mean(np.square(fitness_vector))) \
               * np.mean(fitness_vector * fitness_partials, axis=1)
