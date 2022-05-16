# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import warnings
import pytest
import numpy as np

from bingo.symbolic_regression.equation import Equation
from bingo.symbolic_regression.atomic_potential_regression import PairwiseAtomicPotential, \
                                            PairwiseAtomicTrainingData


class SampleTrainingData:
    def __init__(self, r, potential_energy, config_lims_r):
        self.r = r
        self.potential_energy = potential_energy
        self.config_lims_r = config_lims_r


class SumEquation(Equation):
    def evaluate_equation_at(self, x):
        return np.sum(x, axis=1).reshape((-1, 1))

    def evaluate_equation_with_x_gradient_at(self, x):
        x_sum = self.evaluate_equation_at(x)
        return x_sum, x

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        pass

    def get_complexity(self):
        pass

    def get_latex_string(self):
        pass

    def get_console_string(self):
        pass

    def __str__(self):
        pass

    def distance(self, _chromosome):
        return 0


@pytest.fixture()
def dummy_sum_equation():
    return SumEquation()


@pytest.fixture()
def dummy_training_data():
    r = np.ones((10, 1))
    potential_energy = np.arange(1, 5)
    config_lims_r = [0, 1, 3, 6, 10]
    return SampleTrainingData(r, potential_energy, config_lims_r)


@pytest.fixture()
def two_atom_configuration():
    structure = np.array([[0, 0, 0],
                          [1, 0, 0]])
    period = 1.5
    cuttoff = 0.6
    return structure, period, cuttoff


@pytest.fixture()
def three_atom_configuration():
    structure = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0.7, 1, 0]])
    period = 1.5
    cuttoff = 0.6
    return structure, period, cuttoff


@pytest.fixture()
def six_atom_configuration():
    structure = np.array([[0.7, 1, 0],
                          [0, 0, 0],
                          [1, 0, 0],
                          [0.7, 0.7, 0.7],
                          [0, 0, 1],
                          [0, 0, 0]])
    period = 1.5
    cuttoff = 0.6
    return structure, period, cuttoff


@pytest.fixture()
def configuration_set(two_atom_configuration, three_atom_configuration,
                      six_atom_configuration):
    return [two_atom_configuration, three_atom_configuration,
            six_atom_configuration]


def test_pairwise_potential_regression(dummy_sum_equation,
                                       dummy_training_data):
    regressor = PairwiseAtomicPotential(dummy_training_data)
    fitness = regressor(dummy_sum_equation)
    np.testing.assert_almost_equal(fitness, 0)


def test_reshaping_of_training_data_energies():
    energies = np.ones((1, 1, 3))
    r_list = np.ones((3, 1))
    config_lims = np.arange(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_data = PairwiseAtomicTrainingData(potential_energy=energies,
                                                   r_list=r_list,
                                                   config_lims_r=config_lims)
    assert training_data.potential_energy.ndim == 1


def test_error_training_data_energies_dont_match_configs(
        configuration_set):
    energies = np.ones(2)
    with pytest.raises(ValueError):
        _ = PairwiseAtomicTrainingData(potential_energy=energies,
                                       configurations=configuration_set)


def test_error_if_not_enough_info_for_training_data():
    energies = np.ones(2)
    with pytest.raises(RuntimeError):
        _ = PairwiseAtomicTrainingData(potential_energy=energies)


def test_training_data_synthesis_of_configurations(configuration_set):
    energies = np.ones(3)
    training_data = \
        PairwiseAtomicTrainingData(potential_energy=energies,
                                   configurations=configuration_set)
    expected_r = np.full((9, 1), 0.5)
    expected_r[2:4] = np.sqrt(0.34)
    expected_r[6] = 0.0
    expected_config_lims = [0, 1, 3, 9]
    np.testing.assert_array_almost_equal(training_data.r, expected_r)
    np.testing.assert_array_almost_equal(training_data.config_lims_r,
                                         expected_config_lims)


def test_training_data_subset():
    energies = np.ones(3)
    r_list = np.ones((3, 1))
    config_lims = np.arange(4)
    training_data = PairwiseAtomicTrainingData(potential_energy=energies,
                                               r_list=r_list,
                                               config_lims_r=config_lims)
    subset = training_data[[1, 2]]
    np.testing.assert_array_almost_equal(subset.potential_energy,
                                         np.ones(2))
    np.testing.assert_array_almost_equal(subset.r,
                                         np.ones((2, 1)))
    np.testing.assert_array_almost_equal(subset.config_lims_r,
                                         np.arange(3))


@pytest.mark.parametrize("data_size", [2, 10, 50])
def test_training_data_length(data_size):
    energies = np.ones(data_size)
    r_list = np.ones((data_size, 1))
    config_lims = np.arange(data_size + 1)
    training_data = PairwiseAtomicTrainingData(potential_energy=energies,
                                               r_list=r_list,
                                               config_lims_r=config_lims)
    assert len(training_data) == data_size
