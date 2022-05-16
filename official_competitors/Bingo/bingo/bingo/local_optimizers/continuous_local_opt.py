"""Local optimization of continuous, real-valued parameters

This module contains the implementation of local optimization or continuous,
real-valued parameters in a chromosome.  The local optimization algorithm is
defined as well as the interface that must implemented by chromosomes wishing
to use the functionality.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import optimize

from ..evaluation.fitness_function import FitnessFunction
from ..evaluation.gradient_mixin import GradientMixin, VectorGradientMixin

ROOT_SET = {
    # 'hybr',
    'lm'
    # 'broyden1',
    # 'broyden2',
    # 'anderson',
    # 'linearmixing',
    # 'diagbroyden',
    # 'excitingmixing',
    # 'krylov',
    # 'df-sane'
}

MINIMIZE_SET = {
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    'TNC',
    # 'COBYLA',
    'SLSQP'
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
}

JACOBIAN_SET = {
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'SLSQP',
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov',
    # 'hybr',
    'lm'
}


class ContinuousLocalOptimization(FitnessFunction):
    """Fitness evaluation metric for individuals.

    A class for the fitness evaluation metric of genetic individuals that may
    or may not need local optimization before evaluation.

    Parameters
    ----------
    fitness_function : `FitnessFunction`
        A FitnessFunction that evaluates the fitness of a `chromosomes` in
        bingo. For certain algorithms, `VectorBasedFunction` is required.
        Please see algorithm listing for details.
    algorithm : string
        An algorithm that is used in the local optimization of a
        `chromosomes`. The default algorithm is *Nelder-Mead*. The
        other options are:
            1. FitnessFunction
                - Nelder-Mead
                - Powell
                - CG
                - BFGS
                - Newton-CG (not available yet)
                - L-BFGS-B
                - TNC
                - COBYLA (not available yet)
                - SLSQP
                - trust-constr (not available yet)
                - dogleg (not available yet)
                - trust-ncg (not available yet)
                - trust-exact (not available yet)
                - trust-krylov (not available yet)
            2. VectorBasedFunction
                - hybr (not available yet)
                - lm
                - broyden1 (not available yet)
                - broyden2 (not available yet)
                - anderson (not available yet)
                - linearmixing (not available yet)
                - diagbroyden (not available yet)
                - excitingmixing (not available yet)
                - krylov (not available yet)
                - df-sane (not available yet)

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed by the wrapped
        fitness function
    training_data :
        (Optional) data that can be used in the wrapped fitness function
    param_init_bounds : iterable
        (Optional) Bounds that are used to initialize clo params,
           should be formatted as an iterable [low, high)
           where low will be included in the initialization range and
           high will be excluded
    optimization_options_kwargs :
        (Optional) Keyword arguments for clo options
        e.g. (..., tol=1e-8, options={"maxiter": 1000})
            alternatively you can also do
            (..., **{"tol": 1e-8, "options": {"maxiter": 1000}})

    Raises
    ------
    KeyError:
        `algorithm` must be an algorithm provided by the interface
    TypeError :
        `fitness_function` must Be a valid `FitnessFunction` for the specified
        algorithm
    """
    def __init__(self, fitness_function, algorithm='Nelder-Mead',
                 param_init_bounds=None, **optimization_options_kwargs):
        self._check_algorithm_is_valid(algorithm)
        self._check_root_alg_returns_vector(fitness_function, algorithm)
        self._fitness_function = fitness_function
        self._algorithm = algorithm

        if param_init_bounds is None:
            self.param_init_bounds = [-10000, 10000]
        else:
            self.param_init_bounds = param_init_bounds

        self.optimization_options = optimization_options_kwargs  # default case
        # handled in setter

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self._fitness_function.training_data

    @training_data.setter
    def training_data(self, value):
        self._fitness_function.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._fitness_function.eval_count = value

    @property
    def param_init_bounds(self):
        """[low, high) bounds used to initialize clo params"""
        return self._param_init_bounds

    @param_init_bounds.setter
    def param_init_bounds(self, value):
        self._param_init_bounds = value

    @property
    def optimization_options(self):
        """Continuous local optimization options (e.g. tolerance)"""
        return self._optimization_options

    @optimization_options.setter
    def optimization_options(self, value):
        self._optimization_options = {"tol": 1e-6}
        self._optimization_options.update(**value)

    def __call__(self, individual):
        """Evaluates the fitness of the individual. Provides local optimization
        on `MultipleFloatChromosome` individual if necessary.

        Parameters
        ----------
        individual : `MultipleValueChromosome`
            Individual to which to calculate the fitness. If the individual is
            an instance of `MultipleFloatChromosome`, then local optimization
            may be performed if necessary and the correct `FitnessFunction` is
            provided.

        Returns
        -------
        float :
            The fitness of the invdividual
        """
        if individual.needs_local_optimization():
            self._optimize_params(individual)
        return self._evaluate_fitness(individual)

    @staticmethod
    def _check_algorithm_is_valid(algorithm):
        if algorithm not in ROOT_SET and algorithm not in MINIMIZE_SET:
            raise KeyError("{} is not a listed algorithm".format(algorithm))

    @staticmethod
    def _check_root_alg_returns_vector(fitness_function, algorithm):
        if algorithm in ROOT_SET and not hasattr(fitness_function,
                                                 'evaluate_fitness_vector'):
            raise TypeError("{} requires VectorBasedFunction\
                            as a fitness function".format(algorithm))

    def _optimize_params(self, individual):
        num_params = individual.get_number_local_optimization_params()
        c_0 = np.random.uniform(*self._param_init_bounds, num_params)
        params = self._run_algorithm_for_optimization(
            self._sub_routine_for_fit_function, individual, c_0)
        individual.set_local_optimization_params(params)

    def _sub_routine_for_fit_function(self, params, individual):
        individual.set_local_optimization_params(params)

        if self._algorithm in ROOT_SET:
            if isinstance(self._fitness_function, VectorGradientMixin) \
                    and self._algorithm in JACOBIAN_SET:
                return self._fitness_function.get_fitness_vector_and_jacobian(individual)
            return self._fitness_function.evaluate_fitness_vector(individual)
        return self._fitness_function(individual)

    def _run_algorithm_for_optimization(self, sub_routine, individual, params):
        try:
            if self._algorithm in ROOT_SET:
                if isinstance(self._fitness_function, VectorGradientMixin) \
                        and self._algorithm in JACOBIAN_SET:
                    optimize_result = optimize.root(
                            sub_routine, params,
                            args=(individual, ),
                            jac=True,
                            # jac=lambda x, indv: self._fitness_function.get_fitness_vector_and_jacobian(indv)[1],  # pylint: disable=line-too-long
                            method=self._algorithm,
                            **self._optimization_options)
                else:
                    optimize_result = optimize.root(sub_routine, params,
                                                    args=(individual),
                                                    method=self._algorithm,
                                                    **self._optimization_options)
            else:
                if isinstance(self._fitness_function, GradientMixin) \
                        and self._algorithm in JACOBIAN_SET:
                    optimize_result = optimize.minimize(
                            sub_routine, params,
                            args=(individual, ),
                            jac=lambda x, indv: self._fitness_function.get_fitness_and_gradient(indv)[1],  # pylint: disable=line-too-long
                            method=self._algorithm,
                            **self._optimization_options)
                else:
                    optimize_result = optimize.minimize(sub_routine, params,
                                                        args=(individual, ),
                                                        method=self._algorithm,
                                                        **self._optimization_options)  # pylint: disable=line-too-long
            return optimize_result.x
        except TypeError:
            individual._fitness = np.inf
            individual._fit_set = True
            return params

    def _evaluate_fitness(self, individual):
        return self._fitness_function(individual)


class ChromosomeInterface(metaclass=ABCMeta):
    """For chromosomes with continuous local optimization

    An interface to be used on chromosomes that will be using continuous local
    optimization.
    """
    @abstractmethod
    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        raise NotImplementedError

    @abstractmethod
    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of paramneters to be optimized
        """
        raise NotImplementedError

    @abstractmethod
    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        raise NotImplementedError
