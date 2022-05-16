"""The base of equation chromosomes in bingo.

This module defines the basis of equations in bingo evolutionary analyses.
Equations are commonly used in symbolic regression, a specific application of
genetic programming.
"""
from abc import ABCMeta, abstractmethod

from ..chromosomes.chromosome import Chromosome


class Equation(Chromosome, metaclass=ABCMeta):
    """Base representation of an equation

    This class is the base of a equations used in symbolic regression analyses
    in bingo.
    """

    @abstractmethod
    def evaluate_equation_at(self, x):
        """Evaluate the equation.

        Get value of the equation at points x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        Mx1 array of numeric
            :math:`f(x)`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate equation and get its derivatives.

        Get value the equation at x and its gradient with respect to x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxD array of numeric)
            :math:`f(x)` and :math:`df(x)/dx_i`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate equation and get its derivatives.

        Get value the equation at x and its gradient with respect to
        optimization parameters.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`. L is the number of
            optimization paremeters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_complexity(self):
        """Calculate complexity of equation.

        Returns
        -------
        numeric
            complexity measure of equation
        """
        raise NotImplementedError
