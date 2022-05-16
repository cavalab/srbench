"""
This module contains the basic structure for evolutionary optimization in
bingo.  The general framework allows access to an evolve_until_convergence
function.
"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import os
import logging
from datetime import datetime

import dill

from ..util.argument_validation import argument_validation
from ..util.log import INFO, DETAILED_INFO

LOGGER = logging.getLogger(__name__)
STATS_LOGGER = logging.LoggerAdapter(LOGGER, extra={"stats": True})

OptimizeResult = namedtuple('OptimizeResult', ['success', 'status', 'message',
                                               'ngen', 'fitness', 'time',
                                               'ea_diagnostics'])


class EvolutionaryOptimizer(metaclass=ABCMeta):
    """ Fundamental bingo object that coordinates evolutionary optimization

    Abstract base class for evolutionary optimization.  The primary role of
    this class is to house the evolve_until_convergence function. Classes which
    extend this one will have access to this function's capability.

    Parameters
    ----------
    hall_of_fame : HallOfFame (optional)
        The hall of fame object to be used for storing best individuals

    Attributes
    ----------
    generational_age: int
        The number of generations the optimizer has been evolved
    hall_of_fame: `HallOfFame`
        (optional) An object containing the best individuals seen in the
        optimization
    test_function: `FitnessFunction`
        (optional) A function which can judges the fitness of an individual,
        independent of the `FitnessFunction` used in evolution
    """
    def __init__(self, hall_of_fame=None, test_function=None):
        self.generational_age = 0
        self._starting_age = 0
        self._fitness_improvement_age = 0
        self._best_fitness = None
        self.hall_of_fame = hall_of_fame
        self._previous_checkpoints = []
        self._test_function = test_function
        self._log_stats_header()

    def _log_stats_header(self):
        header = "#generational_age, elapsed_time, " + \
                 "fitness_evaluation_count, current_training_fitness, "
        if self._test_function is not None:
            header += "current_test_fitness, "
        header += "hall_of_fame_fitnesses"
        STATS_LOGGER.log(INFO, header)

    @argument_validation(max_generations={">=": 1},
                         min_generations={">=": 0},
                         convergence_check_frequency={">": 0})
    def evolve_until_convergence(self, max_generations,
                                 fitness_threshold,
                                 convergence_check_frequency=1,
                                 min_generations=0,
                                 stagnation_generations=None,
                                 max_fitness_evaluations=None,
                                 max_time=None,
                                 checkpoint_base_name=None,
                                 num_checkpoints=None):
        """Evolution occurs until one of four convergence criteria is met

        Convergence criteria:
          * a maximum number of generations have been evolved
          * a fitness below an absolute threshold has been achieved
          * improvement upon best fitness has not occurred for a set number of
            generations
          * the maximum number of fitness function evaluations has been reached

        Parameters
        ----------
        max_generations: int
            The maximum number of generations the optimization will run.
        fitness_threshold: float
            The minimum fitness that must be achieved in order for the
            algorithm to converge.
        convergence_check_frequency: int, default 1
            The number of generations that will run between checking for
            convergence.
        min_generations: int, default 0
            The minimum number of generations the algorithm will run.
        stagnation_generations: int
            (optional) The number of generations after which evolution will
            stop if no improvement is seen.
        max_fitness_evaluations: int
            (optional) The maximum number of fitness function evaluations
            (approximately) the optimizer will run.
        max_time : float
            (optional) The time limit for the evolution, in seconds.
        checkpoint_base_name: str
            (optional) base file name for checkpoint files
        num_checkpoints: int
            (optional) number of recent checkpoints to keep, previous ones are
            removed

        Returns
        --------
        `OptimizeResult` :
            Object containing information about the result of the evolution
        """
        start_time = datetime.now()
        self._starting_age = self.generational_age
        self._update_best_fitness()
        self._update_checkpoints(checkpoint_base_name, num_checkpoints,
                                 reset=True)
        self._log_optimization(start_time)

        while self.generational_age - self._starting_age < min_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()
            self._update_checkpoints(checkpoint_base_name, num_checkpoints)
            self._log_optimization(start_time)

        _exit, result = self._check_exit_criteria(fitness_threshold,
                                                  stagnation_generations,
                                                  max_fitness_evaluations,
                                                  max_time,
                                                  start_time)
        if _exit:
            self._log_exit(result)
            return result

        while self.generational_age - self._starting_age < max_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()
            self._update_checkpoints(checkpoint_base_name, num_checkpoints)
            self._log_optimization(start_time)

            _exit, result = self._check_exit_criteria(fitness_threshold,
                                                      stagnation_generations,
                                                      max_fitness_evaluations,
                                                      max_time,
                                                      start_time)
            if _exit:
                self._log_exit(result)
                return result

        result = self._make_optim_result(2, start_time, max_generations)
        self._log_exit(result)
        return result

    def _log_optimization(self, start_time):
        test_fitness = None
        if self._test_function is not None:
            test_fitness = self._test_function(self.get_best_individual())
        log_string = "Generation: %d \t " % self.generational_age
        elapsed_time = datetime.now() - start_time
        log_string += "Elapsed time: %s \t " % elapsed_time
        log_string += "Best training fitness: %le \t " % self._best_fitness
        if test_fitness is not None:
            log_string += "Test fitness: %le \t " % test_fitness
        LOGGER.log(INFO, log_string)

        stats_string = "%d, %le, %d, %le" % \
                       (self.generational_age, elapsed_time.total_seconds(),
                        self.get_fitness_evaluation_count(),
                        self._best_fitness)
        if test_fitness is not None:
            stats_string += ", %le" % test_fitness
        if self.hall_of_fame is not None:
            for i in self.hall_of_fame:
                stats_string += ", %le" % i.fitness
        STATS_LOGGER.log(INFO, stats_string)

    def _update_best_fitness(self):
        last_best_fitness = self._best_fitness
        self._best_fitness = self.get_best_fitness()
        if last_best_fitness is None or self._best_fitness < last_best_fitness:
            self._fitness_improvement_age = self.generational_age

    def _update_checkpoints(self, checkpoint_base_name, num_checkpoints,
                            reset=False):
        if reset:
            self._previous_checkpoints = []

        if checkpoint_base_name is not None:
            checkpoint_file_name = "{}_{}.pkl".format(checkpoint_base_name,
                                                      self.generational_age)
            self.dump_to_file(checkpoint_file_name)
            if num_checkpoints is not None:
                self._previous_checkpoints.append(checkpoint_file_name)
                if len(self._previous_checkpoints) > num_checkpoints:
                    self._remove_stale_checkpoint()

    def _remove_stale_checkpoint(self):
        LOGGER.debug("Removing stale checkpoint file: %s",
                     self._previous_checkpoints[0])
        os.remove(self._previous_checkpoints.pop(0))

    def _check_exit_criteria(self, fitness_threshold, stagnation_generations,
                             max_fitness_evaluations, max_time, start_time):
        if self._convergence(fitness_threshold):
            return True, self._make_optim_result(0, start_time,
                                                 fitness_threshold)
        if self._stagnation(stagnation_generations):
            return True, self._make_optim_result(1, start_time,
                                                 stagnation_generations)
        if self._hit_max_evals(max_fitness_evaluations):
            return True, self._make_optim_result(3, start_time,
                                                 max_fitness_evaluations)
        if self._hit_time_limit(max_time, start_time):
            return True, self._make_optim_result(4, start_time, max_time)
        return False, None

    def _convergence(self, threshold):
        return self._best_fitness <= threshold

    def _stagnation(self, threshold):
        if threshold is None:
            return False
        stagnation_time = self.generational_age - self._fitness_improvement_age
        return stagnation_time >= threshold

    def _hit_max_evals(self, threshold):
        if threshold is None:
            return False
        return self.get_fitness_evaluation_count() >= threshold

    @staticmethod
    def _hit_time_limit(threshold, start_time):
        if threshold is None:
            return False
        run_time = (datetime.now() - start_time).total_seconds()
        return run_time >= threshold

    def _make_optim_result(self, status, start_time, aux_info):
        ngen = self.generational_age - self._starting_age
        run_time = (datetime.now() - start_time).total_seconds()
        ea_diagnostics = self.get_ea_diagnostic_info().summary
        if status == 0:
            message = "Absolte convergence occurred with best fitness < " + \
                      "{}".format(aux_info)
            success = True
        elif status == 1:
            message = "Stagnation occurred with no improvement for more " + \
                      "than {} generations".format(aux_info)
            success = False
        elif status == 2:
            message = "The maximum number of generational steps " + \
                      "({}) occurred".format(aux_info)
            success = False
        elif status == 3:
            message = "The maximum number of fitness evaluations " + \
                      "({}) was exceeded. Total fitness ".format(aux_info) + \
                      "evals: {}".format(self.get_fitness_evaluation_count())
            success = False
        else:  # status == 4:
            message = "The maximum time ({}) was exceeded.".format(aux_info)
            success = False
        return OptimizeResult(success, status, message, ngen,
                              self._best_fitness, run_time, ea_diagnostics)

    def _log_exit(self, result):
        if result.success:
            LOGGER.log(INFO, "Evolution successfully converged.")
        else:
            LOGGER.log(INFO, "Evolution unsuccessful.")
        LOGGER.log(INFO, "  %s", result.message)
        if self.hall_of_fame is not None:
            LOGGER.log(INFO, "Hall of Fame:\n%s", self.hall_of_fame)

    def evolve(self, num_generations, hall_of_fame_update=True,
               suppress_logging=False):
        """The function responsible for generational evolution.

        Parameters
        ----------
        num_generations : int
            The number of generations to evolve
        hall_of_fame_update : bool (optional)
            Used to manually turn on or off the hall of fame update. Default
            True.
        suppress_logging : bool (optional)
            Used to manually suppress the logging output of this function
        """
        start_time = datetime.now()
        self._do_evolution(num_generations)
        if hall_of_fame_update:
            self.update_hall_of_fame()
        if not suppress_logging:
            self._log_evolution(start_time)

    def _log_evolution(self, start_time):
        elapsed_time = datetime.now() - start_time
        LOGGER.log(DETAILED_INFO, "Evolution time %s\t fitness %.3le",
                   elapsed_time, self.get_best_fitness())

    def update_hall_of_fame(self):
        """Manually update the hall of fame"""
        if self.hall_of_fame is not None:
            self.hall_of_fame.update(self._get_potential_hof_members())
            LOGGER.debug("Hall of fame updated")

    @abstractmethod
    def _do_evolution(self, num_generations):
        """Definition of this function should do the heavy lifting of
        performing evolutionary development.

        Parameters
        ----------
        num_generations : int
            The number of generations to evolve

        Notes
        -----
        This function is responsible for incrementing generational age
        """
        raise NotImplementedError

    @abstractmethod
    def _get_potential_hof_members(self):
        """Definition of this function should return the individuals which
        should be considered for induction into the hall of fame.

        Returns
        ----------
        list of chromosomes :
            Potential hall of fame members
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_individual(self):
        """ Gets the most fit individual

        Returns
        -------
        chromosomes :
            Best individual
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_fitness(self):
        """ Gets the fitness value of the most fit individual

        Returns
        -------
         :
            Fitness of best individual
        """
        raise NotImplementedError

    @abstractmethod
    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        raise NotImplementedError

    @abstractmethod
    def get_ea_diagnostic_info(self):
        """ Gets diagnostic info from the evolutionary algorithm(s)

        Returns
        -------
        EaDiagnosticsSummary :
            summary of evolutionary algorithm diagnostics
        """
        raise NotImplementedError

    def dump_to_file(self, filename):
        """ Dump the evolutionary_optimizers object to a pickle file

        Parameters
        ----------
        filename : str
            the name of the pickle file to dump
        """
        LOGGER.log(INFO, "Saving checkpoint: %s", filename)
        with open(filename, "wb") as dump_file:
            dill.dump(self, dump_file, protocol=dill.HIGHEST_PROTOCOL)
        LOGGER.log(DETAILED_INFO, "Saved successfully")


def load_evolutionary_optimizer_from_file(filename):
    """ Load an evolutionary_optimizers object from a pickle file

    Parameters
    ----------
    filename : str
        the name of the pickle file to load

    Returns
    -------
    str :
        an evolutionary optimizer
    """
    LOGGER.log(INFO, "Loading checkpoint file: %s", filename)
    with open(filename, "rb") as load_file:
        ev_opt = dill.load(load_file)
    LOGGER.log(DETAILED_INFO, "Loaded successfully")
    return ev_opt
