"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import numpy as np

try:
    from bingocpp import AGraph
    BINGOCPP = True
except (ImportError, KeyError, ModuleNotFoundError) as e:
    from .agraph import AGraph
    BINGOCPP = False
from .agraph import AGraph as pyAGraph
from .agraph import force_use_of_python_simplification
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation


class AGraphGenerator(Generator):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : agraph.ComponentGenerator
                          Generator of stack components of agraphs
    """
    @argument_validation(agraph_size={">=": 1})
    def __init__(self, agraph_size, component_generator, use_python=False,
                 use_simplification=False):
        self.agraph_size = agraph_size
        self.component_generator = component_generator
        self._use_simplification = use_simplification
        if use_python:
            self._backend_generator_function = self._python_generator_function
        else:
            self._backend_generator_function = self._generator_function

        if use_simplification:
            force_use_of_python_simplification()

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        individual = self._backend_generator_function()
        individual.command_array = self._create_command_array()
        return individual

    def _python_generator_function(self):
        return pyAGraph(use_simplification=self._use_simplification)

    def _generator_function(self):
        return AGraph(use_simplification=self._use_simplification)

    def _create_command_array(self):
        command_array = np.empty((self.agraph_size, 3), dtype=int)
        for i in range(self.agraph_size):
            command_array[i] = self.component_generator.random_command(i)
        return command_array
