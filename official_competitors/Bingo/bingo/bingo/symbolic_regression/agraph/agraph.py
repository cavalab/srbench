"""Acyclic graph representation of an equation.


This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression.

Stack
-----

The stack is represented as Nx3 integer array. Where each row of the array
corresponds to a single command with form:

========  ===========  ===========
Node      Parameter 1  Parameter 2
========  ===========  ===========

Where the parameters are a reference to the result of previously executed
commands (referenced by row number in the stack). The result of the last (N'th)
command in the stack is the evaluation of the equation.

Note: Parameter values have special meaning for two of the nodes (0 and 1).

Nodes
-----

An integer to node mapping is how the command stack is parsed.
The current map is outlined below.

========  =======================================  =================
Node      Name                                     Math
========  =======================================  =================
0         load p1'th column of x                   :math:`x_{p1}`
1         load p1'th constant                      :math:`c_{p1}`
2         addition                                 :math:`p1 + p2`
3         subtraction                              :math:`p1 - p2`
4         multiplication                           :math:`p1 - p2`
5         division (not divide-by-zero protected)  :math:`p1 / p2`
6         sine                                     :math:`sin(p1)`
7         cosine                                   :math:`cos(p1)`
8         exponential                              :math:`exp(p1)`
9         logarithm                                :math:`log(|p1|)`
10        power                                    :math:`p1^{p2}`
11        absolute value                           :math:`|p1|`
12        square root                              :math:`sqrt(|p1|)`
13        safe                                     :math:`|p1|^{p2}`
14        hyperbolic sine                          :math:`sinh(p1)`
15        hyperbolic cosine                        :math:`cosh(p1)`
========  =======================================  =================
"""
import logging
import numpy as np

from .string_generation import get_formatted_string
from ..equation import Equation
from ...local_optimizers import continuous_local_opt
from .operator_definitions import CONSTANT

try:
    from bingocpp import evaluation_backend
    from bingocpp import simplification_backend
except ImportError:
    from .evaluation_backend import evaluation_backend
    from .simplification_backend import simplification_backend

LOGGER = logging.getLogger(__name__)

USING_PYTHON_SIMPLIFICATION = False


def force_use_of_python_backends():
    """When c++ backends are available, this can be used to force the use of
    python backends"""
    # pylint: disable=redefined-outer-name, global-statement, invalid-name
    # pylint: disable=import-outside-toplevel
    global evaluation_backend, simplification_backend
    from .evaluation_backend import evaluation_backend
    from .simplification_backend import simplification_backend


def force_use_of_python_simplification():
    """When c++ simplification is available, this can be used to force the use
    of python simplification"""
    # pylint: disable=redefined-outer-name, global-statement, invalid-name
    # pylint: disable=import-outside-toplevel
    global simplification_backend
    from .simplification_backend import simplification_backend

    global USING_PYTHON_SIMPLIFICATION
    USING_PYTHON_SIMPLIFICATION = True


class AGraph(Equation, continuous_local_opt.ChromosomeInterface):
    """Acyclic graph representation of an equation.

    `AGraph` is initialized with with empty command array and no constants.

    Attributes
    ----------
    command_array : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack. This is read-only.
    mutable_command_array : Nx3 numpy array of int.
        The version of the command array that must be used if modifications are
        to be made.
    constants : tuple of numeric
        numeric constants that are used in the equation

    """
    def __init__(self, use_simplification=False):
        super().__init__()
        self._command_array = np.empty([0, 3], dtype=int)

        self._simplified_command_array = np.empty([0, 3], dtype=int)
        self._simplified_constants = []

        self._needs_opt = False
        self._modified = False
        self._use_simplification = use_simplification

        if use_simplification and not USING_PYTHON_SIMPLIFICATION:
            force_use_of_python_simplification()

    @property
    def engine(self):
        """Identification of the code location"""
        return "Python"

    @property
    def command_array(self):
        """Nx3 array of int: acyclic graph stack.

        Notes
        -----
        Setting the command stack automatically resets fitness
        """
        self._command_array.flags.writeable = False
        return self._command_array

    @command_array.setter
    def command_array(self, command_array):
        self._command_array = command_array
        self._notify_modification()

    @property
    def mutable_command_array(self):
        """Nx3 array of int: acyclic graph stack.

        Notes
        -----
        Setting the command stack automatically resets fitness
        """
        self._command_array.flags.writeable = True
        self._notify_modification()
        return self._command_array

    @property
    def constants(self):
        """The numerical constants in the equation."""
        return self._simplified_constants

    def _notify_modification(self):
        self._modified = True
        self._fitness = None
        self._fit_set = False

    def _update(self):
        if self._use_simplification:
            self._simplified_command_array = \
                simplification_backend.simplify_stack(self._command_array)
        else:
            self._simplified_command_array = \
                simplification_backend.reduce_stack(self._command_array)

        const_commands = self._simplified_command_array[:, 0] == CONSTANT
        num_const = np.count_nonzero(const_commands)
        self._simplified_command_array[const_commands, 1] = np.arange(num_const)
        self._simplified_command_array[const_commands, 2] = np.arange(num_const)

        optimization_aggression = 0
        if optimization_aggression == 0 \
                and num_const <= len(self._simplified_constants):
            self._simplified_constants = self._simplified_constants[:num_const]
        elif optimization_aggression == 1 \
                and num_const == len(self._simplified_constants):
            self._simplified_constants = self._simplified_constants[:num_const]
        else:
            self._simplified_constants = (1.0,) * num_const
            if num_const > 0:
                self._needs_opt = True
        self._modified = False

    def needs_local_optimization(self):
        """The `AGraph` needs local optimization.

        Find out whether constants need optimization.

        Returns
        -------
        bool :
            Constants need optimization
        """
        if self._modified:
            self._update()
        return self._needs_opt

    def get_number_local_optimization_params(self):
        """number of parameters for local optimization

        Count constants and set up for optimization

        Returns
        -------
        int :
            Number of constants that need to be optimized
        """
        if self._modified:
            self._update()
        return len(self._simplified_constants)

    def set_local_optimization_params(self, params):
        """Set the local optimization parameters.

        Manually set optimized constants.

        Parameters
        ----------
        params : list of numeric
            Values to set constants
        """
        self._simplified_constants = tuple(params)
        self._needs_opt = False

    def get_utilized_commands(self):
        """"Find which commands are utilized.

        Find the commands in the command array of the `AGraph` upon which the
        last command relies. This is inclusive of the last command.

        Returns
        -------
        list of bool of length N
            Boolean values for whether each command is utilized.
        """
        return simplification_backend.get_utilized_commands(
            self._command_array)

    def evaluate_equation_at(self, x):
        """Evaluate the `AGraph` equation.

        evaluation of the `AGraph` equation at points x.

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
        if self._modified:
            self._update()
        try:
            f_of_x = \
                evaluation_backend.evaluate(self._simplified_command_array, x,
                                            self._simplified_constants)
            return f_of_x
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation", err)
            return np.full(x.shape, np.nan)

    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate `AGraph` and get its derivatives.

        Evaluate the `AGraph` equation at x and the gradient of x.

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
        if self._modified:
            self._update()
        try:
            f_of_x, df_dx = evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants, True)
            return f_of_x, df_dx
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/deriv", err)
            nan_array = np.full(x.shape, np.nan)
            return nan_array, np.array(nan_array)

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate `AGraph` and get its derivatives.

        Evaluate the `AGraph` equation at x and get the gradient of constants.
        Constants are of length L.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dc = evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants, False)
            return f_of_x, df_dc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_array = np.full((x.shape[0], len(self._simplified_constants)),
                                np.nan)
            return nan_array, np.array(nan_array)

    def __str__(self):
        """Console string output of `AGraph` equation.

        Returns
        -------
        str :
            equation in console form
        """
        return self.get_formatted_string("console")

    def get_formatted_string(self, format_, raw=False):
        """Output a string description of the the `AGraph` in a given format.

        Parameters
        ----------
        format_ : str
            The requested format of the equation. Options are "console",
            "latex", and "stack".
        raw : bool
            (optional) Output of the raw command array rather than the
            processed version. Default False.

        Returns
        -------
        str :
            Equation in specified form
        """
        if raw:
            return get_formatted_string(format_, self._command_array, tuple())
        if self._modified:
            self._update()
        return get_formatted_string(format_, self._simplified_command_array,
                                    self._simplified_constants)

    def get_complexity(self):
        """Calculate complexity of agraph equation.

        Returns
        -------
        int :
            number of utilized commands in stack
        """
        if self._modified:
            self._update()
        return self._simplified_command_array.shape[0]

    def distance(self, chromosome):
        """Computes the distance to another `AGraph`

        Distance is a measure of similarity of the two command_arrays

        Parameters
        ----------
        chromosome : `AGraph`
            The individual to which distance will be calculated

        Returns
        -------
        int :
            distance from self to individual
        """
        dist = np.sum(self.command_array != chromosome.command_array)
        return dist

    def __deepcopy__(self, memodict=None):
        duplicate = AGraph()
        self._copy_agraph_values_to_new_graph(duplicate)
        return duplicate

    def _copy_agraph_values_to_new_graph(self, agraph_duplicate):
        # pylint: disable=protected-access
        agraph_duplicate._genetic_age = self._genetic_age
        agraph_duplicate._fitness = self._fitness
        agraph_duplicate._fit_set = self._fit_set
        agraph_duplicate._command_array = np.copy(self.command_array)
        agraph_duplicate._simplified_command_array = \
            np.copy(self._simplified_command_array)
        agraph_duplicate._simplified_constants = \
            tuple(self._simplified_constants)
        agraph_duplicate._needs_opt = self._needs_opt
        agraph_duplicate._modified = self._modified
        agraph_duplicate._use_simplification = self._use_simplification
