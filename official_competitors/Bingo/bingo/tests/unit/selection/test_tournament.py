# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
from bingo.selection.tournament import Tournament
from bingo.chromosomes.chromosome import Chromosome


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_01234(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in range(5)]


@pytest.mark.parametrize("tourn_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_tournament_size(tourn_size, expected_error):
    with pytest.raises(expected_error):
        _ = Tournament(tourn_size)


def test_tournament_selects_best_indv(population_01234):
    tourn = Tournament(tournament_size=5)
    new_population = tourn(population_01234, 1)
    assert new_population[0].fitness == 0


@pytest.mark.parametrize("new_pop_size", range(5))
def test_tournament_returns_correct_size_population(population_01234,
                                                    new_pop_size):
    tourn = Tournament(tournament_size=4)
    new_population = tourn(population_01234, new_pop_size)
    assert len(new_population) == new_pop_size


def test_no_repeats_in_selected_population(population_01234):
    tourn = Tournament(tournament_size=4)
    new_population = tourn(population_01234, 3)
    for i, indv in enumerate(new_population[:-1]):
        assert indv not in new_population[i + 1:]
