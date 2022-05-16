# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import pytest
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evolutionary_optimizers.island import Island
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def dummy_island(mocker):
    mocker.patch('bingo.evolutionary_optimizers.island.Island', autospec=True)
    mocker.patch.object(Island, "__abstractmethods__", new_callable=set)

    def mocked_generator():
        return mocker.Mock()

    mocked_ea = mocker.Mock()
    return Island(mocked_ea, mocked_generator, population_size=10)


@pytest.mark.parametrize("num_islands", range(1, 4))
def test_creation_of_multiple_islands(mocker, num_islands):
    mocked_island = mocker.Mock()
    arch = SerialArchipelago(mocked_island, num_islands)
    assert len(arch.islands) == num_islands
    for island in arch.islands:
        island.regenerate_population.assert_called_once()


def test_step_through_generations(mocker):
    mocked_island = mocker.Mock()
    arch = SerialArchipelago(mocked_island, num_islands=2)
    arch._step_through_generations(123)
    for island in arch.islands:
        island.evolve.assert_called_once_with(123, hall_of_fame_update=False)


def test_migration(dummy_island):
    arch = SerialArchipelago(dummy_island, num_islands=2)
    initial_pop_0 = set(arch.islands[0].population)
    initial_pop_1 = set(arch.islands[1].population)
    arch._coordinate_migration_between_islands()
    final_pop_0 = set(arch.islands[0].population)
    final_pop_1 = set(arch.islands[1].population)

    assert len(final_pop_0.intersection(initial_pop_0)) == 5
    assert len(final_pop_0.intersection(initial_pop_1)) == 5
    assert len(final_pop_1.intersection(initial_pop_0)) == 5
    assert len(final_pop_1.intersection(initial_pop_1)) == 5
    assert final_pop_0.union(final_pop_1) == initial_pop_0.union(initial_pop_1)


@pytest.mark.parametrize("island_with_best", range(4))
def test_best_individual(mocker, dummy_chromosome, island_with_best):
    mocked_island = mocker.Mock()
    mocked_island.get_best_individual.return_value = \
        dummy_chromosome(fitness=1)

    best_indv = dummy_chromosome(fitness=0)
    mocked_island_with_best = mocker.Mock()
    mocked_island_with_best.get_best_individual.return_value = best_indv

    arch = SerialArchipelago(mocked_island, num_islands=4)
    arch.islands[island_with_best] = mocked_island_with_best

    assert arch.get_best_individual() == best_indv


def test_fitness_eval_count(mocker):
    num_islands = 4
    num_evals = 3
    mocked_island = mocker.Mock()
    mocked_island.get_fitness_evaluation_count.return_value = num_evals
    arch = SerialArchipelago(mocked_island, num_islands=num_islands)
    assert arch.get_fitness_evaluation_count() == num_islands * num_evals


def test_ea_diagnostics(mocker):
    num_islands = 4
    diagnostics = 3
    mocked_island = mocker.Mock()
    mocked_island.get_ea_diagnostic_info.return_value = diagnostics
    arch = SerialArchipelago(mocked_island, num_islands=num_islands)
    assert arch.get_ea_diagnostic_info() == num_islands * diagnostics
