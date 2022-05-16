# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import pytest
from collections import namedtuple

try:
    from bingo.evolutionary_optimizers.parallel_archipelago \
        import ParallelArchipelago, load_parallel_archipelago_from_file
    PAR_ARCH_LOADED = True
except ImportError:
    PAR_ARCH_LOADED = False

from bingo.evolutionary_optimizers.island import Island


DummyChromosome = namedtuple("DummyChromosome", ["fitness"])


@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize("sync_frequency", [10, 12])
def test_step_through_generations(mocker, non_blocking, sync_frequency):
    mocked_island = mocker.Mock()
    type(mocked_island).generational_age = \
        mocker.PropertyMock(side_effect=list(range(sync_frequency, 200,
                                                   sync_frequency)))
    arch = ParallelArchipelago(mocked_island, non_blocking=non_blocking,
                               sync_frequency=sync_frequency)
    arch._step_through_generations(120)
    if non_blocking:
        assert mocked_island.evolve.call_count == 120 // sync_frequency
    else:
        mocked_island.evolve.assert_called_once_with(120,
                                                     hall_of_fame_update=False,
                                                     suppress_logging=True)


def test_best_individual(mocker):
    best_indv = DummyChromosome(fitness=0)
    mocked_island_with_best = mocker.Mock()
    mocked_island_with_best.get_best_individual.return_value = best_indv
    arch = ParallelArchipelago(mocked_island_with_best)
    assert arch.get_best_individual() == best_indv


def test_fitness_eval_count(mocker):
    num_evals = 3
    mocked_island = mocker.Mock()
    mocked_island.get_fitness_evaluation_count.return_value = num_evals
    arch = ParallelArchipelago(mocked_island)
    assert arch.get_fitness_evaluation_count() == num_evals


def test_ea_diagnostics(mocker):
    diagnostics = 3
    mocked_island = mocker.Mock()
    mocked_island.get_ea_diagnostic_info.return_value = diagnostics
    arch = ParallelArchipelago(mocked_island)
    assert arch.get_ea_diagnostic_info() == diagnostics
