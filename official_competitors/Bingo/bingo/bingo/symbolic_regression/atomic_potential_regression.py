"""Symbolic Regression of Inter-atomic Potentials

Symbolic regression of inter-atomic potentials is defined in this module as the
search for a function, f, such that sum( f(r_i) ) - U. U is the total potential
energy of an atomic configuration with a set of atoms separated at distances of
r_i.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to symbolic regression of inter-atomic potentials. Namely,
these classes are an appropriate fitness evaluator and a corresponding training
data container.
"""
import warnings
import logging

import numpy as np

from ..evaluation.fitness_function import VectorBasedFunction
from ..evaluation.training_data import TrainingData

LOGGER = logging.getLogger(__name__)


class PairwiseAtomicPotential(VectorBasedFunction):
    """Fitness based on total potential energy of a set of configurations.

    Pairwise atomic potential which is fit with total potential energy for a
    set of configurations. Fitness is calculated as how well total potential
    energies are matched by the summation of pairwise energies which are
    calculated by the Equation individual

    fitness = sum(abs(  sum( f(r_i) ) - U_true_i  ))    for i in config

    Parameters
    ----------
    training_data : PairwiseAtomicTrainingData
                   data that is used in fitness evaluation.  Must have
                   attributes r, potential_energy and config_lims_r.
    """

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        pair_energies = individual.evaluate_equation_at(
            self.training_data.r).flatten()

        err_vec = []
        for i, energy_true in enumerate(self.training_data.potential_energy):
            energy = np.sum(pair_energies[self.training_data.config_lims_r[i]:
                                          self.training_data.config_lims_r[
                                              i + 1]])
            err_vec.append(energy - energy_true)

        return np.array(err_vec).flatten()


class PairwiseAtomicTrainingData(TrainingData):
    """PairwiseAtomicTrainingData:

    Training data of this type contains distances (r) between ataoms in several
    atomic configurations. Each configuration has an associated potential
    energy.  The r values beloning to each configuration are bounded by
    configuration limits (config_lims_r)

    Parameters
    ----------
     potential_energy : 1d numpy array
                        potential energy for each configuration
     configurations : (optional) list of tuples (structure, period, r_cutoff),
                      where the structure is an array of x,y,z locations of
                      atoms. Period is the periodic size of the configuration.
                      rcutoff is the cutoff distance after which the pairwise
                      interaction does not effect the potential energy.
     r_list : 2d numpy array
              (optional) list of all pairwise distances
     config_lims_r : 1d numpy array
                     (optional) bounds of all of the r_indices corresponding to
                     each configuration

    Notes
    -----
    Ininilization must be performed with either configurations or a
    combination of r_list and config_lims_r.
    """
    def __init__(self, potential_energy, configurations=None, r_list=None,
                 config_lims_r=None):

        potential_energy = self._flatten_energies_if_needed(potential_energy)

        if configurations is not None:
            self._check_equal_num_of_energies_and_configs(configurations,
                                                          potential_energy)
            config_lims_r, r_list = \
                self._synthesize_atomic_configurations(configurations)

        elif r_list is None or config_lims_r is None:
            raise RuntimeError('Invalid construction of ' +
                               'PairwiseAtomicTrainingData')

        self.r = r_list
        self.config_lims_r = config_lims_r
        self.potential_energy = potential_energy

    def __getitem__(self, items):
        """gets a subset of the PairwiseAtomicTrainingData

        Parameters
        ----------
         items : list or int
                 index (or indices) of the subset

        Returns
        -------
         PairwiseAtomicTrainingData :
                                       a subset
        """

        r_inds = []
        new_config_lims_r = [0]
        for i in items:
            r_inds += range(self.config_lims_r[i], self.config_lims_r[i+1])
            new_config_lims_r.append(len(r_inds))
        new_config_lims_r = np.array(new_config_lims_r)

        new_potential_energy = self.potential_energy[items]
        temp = PairwiseAtomicTrainingData(
            potential_energy=new_potential_energy,
            r_list=self.r[r_inds, :],
            config_lims_r=new_config_lims_r)
        return temp

    def __len__(self):
        """gets the length of the first dimension of the data

        Returns
        -------
         int :
                index-able size
        """
        return self.potential_energy.shape[0]

    @staticmethod
    def _check_equal_num_of_energies_and_configs(configurations,
                                                 potential_energy):
        if potential_energy.shape[0] != len(configurations):
            raise ValueError("Pairwise atomic training data: potential " +
                             "energy and configurations are different " +
                             "sizes")

    @staticmethod
    def _flatten_energies_if_needed(potential_energy):
        if potential_energy.ndim > 1:
            warnings.warn("Pairwise atomic training data: potential energy " +
                          "should be 1 dim array, flattening array")
            potential_energy = potential_energy.flatten()
        return potential_energy

    @staticmethod
    def _synthesize_atomic_configurations(configurations):
        r_list = []
        config_lims_r = [0]
        for (structure, periodic_size, r_cutoff) in configurations:
            # make radius list
            natoms = structure.shape[0]
            rcutsq = r_cutoff ** 2
            for atomi in range(0, natoms):
                xtmp = structure[atomi, 0]
                ytmp = structure[atomi, 1]
                ztmp = structure[atomi, 2]
                for atomj in range(atomi + 1, natoms):
                    delx = structure[atomj, 0] - xtmp
                    while delx > 0.5 * periodic_size:
                        delx -= periodic_size
                    while delx < -0.5 * periodic_size:
                        delx += periodic_size
                    dely = structure[atomj, 1] - ytmp
                    while dely > 0.5 * periodic_size:
                        dely -= periodic_size
                    while dely < -0.5 * periodic_size:
                        dely += periodic_size
                    delz = structure[atomj, 2] - ztmp
                    while delz > 0.5 * periodic_size:
                        delz -= periodic_size
                    while delz < -0.5 * periodic_size:
                        delz += periodic_size

                    rsq = delx * delx + dely * dely + delz * delz
                    if rsq <= rcutsq:
                        r_list.append(np.sqrt(rsq))
            config_lims_r.append(len(r_list))
        r_list = np.array(r_list).reshape([-1, 1])
        config_lims_r = np.array(config_lims_r)
        return config_lims_r, r_list
