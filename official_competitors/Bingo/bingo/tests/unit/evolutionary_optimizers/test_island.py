# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
from bingo.evolutionary_optimizers.island import Island
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


def test_raises_error_for_negative_pop_size(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    with pytest.raises(ValueError):
        _ = Island(mocked_ea, mocked_generator, population_size=-1)


@pytest.mark.parametrize("pop_size", range(5))
def test_pop_size(mocker, pop_size):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, pop_size)
    assert len(island.population) == pop_size


def test_do_evolution(mocker, dummy_chromosome):
    def dummy_generator():
        return dummy_chromosome()
    mocked_ea = mocker.Mock()
    mocked_ea.generational_step.return_value = \
        [dummy_chromosome() for _ in range(10)]
    island = Island(mocked_ea, dummy_generator, population_size=10)
    island._do_evolution(num_generations=2)

    assert mocked_ea.generational_step.call_count == 2


def test_evaluate_population(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)
    island.evaluate_population()
    mocked_ea.evaluation.assert_called_once()


@pytest.mark.parametrize("best_index", [0, 2, 4])
def test_best_individual(mocker, dummy_chromosome, best_index):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)

    best_indv = dummy_chromosome(fitness=0)
    dummy_pop = [dummy_chromosome(fitness=1) for _ in range(5)]
    island.population = list(dummy_pop)
    island.population[best_index] = best_indv

    assert island.get_best_individual() == best_indv


def test_best_fitness(mocker, dummy_chromosome):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)

    dummy_pop = [dummy_chromosome(fitness=5 - i) for i in range(5)]
    island.population = dummy_pop

    assert island.get_best_fitness() == 1


def test_fitness_eval_pass_through(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    mocked_ea.evaluation.eval_count = 123
    island = Island(mocked_ea, mocked_generator, population_size=10)
    assert island.get_fitness_evaluation_count() == 123


def test_ea_diagnostics_pass_through(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    mocked_ea.diagnostics = 123
    island = Island(mocked_ea, mocked_generator, population_size=10)
    assert island.get_ea_diagnostic_info() == 123


def test_regenerate_population(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)
    initial_pop = island.population
    island.regenerate_population()

    assert island.population is not initial_pop
    assert len(island.population) == len(initial_pop)


@pytest.mark.parametrize("num_dump", range(4))
def test_dump_fraction_of_population(mocker, num_dump):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)

    pop_frac = island.dump_fraction_of_population(fraction=num_dump/10)
    assert len(pop_frac) == num_dump
    assert len(island.population) == 10 - num_dump


def test_hof_members(mocker):
    mocked_generator = mocker.Mock()
    mocked_ea = mocker.Mock()
    island = Island(mocked_ea, mocked_generator, population_size=10)
    assert island._get_potential_hof_members() is island.population
