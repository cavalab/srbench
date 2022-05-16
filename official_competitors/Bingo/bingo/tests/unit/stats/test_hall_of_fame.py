# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np
from bingo.stats.hall_of_fame import HallOfFame


DummyIndv = namedtuple('DummyIndv', ['fitness', 'gene'])


@pytest.fixture
def empty_hof():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    return HallOfFame(5, similarity_function=similar)


@pytest.fixture
def partial_hof():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    partial_hof = HallOfFame(5, similarity_function=similar)
    for i in range(3):
        partial_hof.insert(DummyIndv(i, i))
    return partial_hof


@pytest.fixture
def full_hof():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    full_hof =  HallOfFame(5, similarity_function=similar)
    for i in range(5):
        full_hof.insert(DummyIndv(i, i))
    return full_hof


@pytest.fixture(params=["empty", "partial", "full"])
def all_hofs(request, empty_hof, partial_hof, full_hof):
    if request.param == "empty":
        return empty_hof
    if request.param == "partial":
        return partial_hof
    return full_hof


def test_access_hof_like_list(partial_hof):
    assert partial_hof[1].fitness == 1
    assert partial_hof[-1].fitness == 2

    for i, indv in enumerate(partial_hof):
        assert indv.fitness == i

    for i, indv in enumerate(reversed(partial_hof)):
        assert indv.fitness == 2 - i


def test_string_output(partial_hof):
    expected_string = """0\tDummyIndv(fitness=0, gene=0)
1\tDummyIndv(fitness=1, gene=1)
2\tDummyIndv(fitness=2, gene=2)"""
    assert str(partial_hof) == expected_string


def test_len_of_hof(partial_hof):
    assert len(partial_hof) == 3


def test_remove_item_from_hof(partial_hof):
    partial_hof.remove(1)
    for indv, expected_fitness in zip(partial_hof, [0, 2]):
        assert indv.fitness == pytest.approx(expected_fitness)


def test_clear_hof(partial_hof):
    partial_hof.clear()
    assert len(partial_hof) == 0


def test_insert_into_partial_hof(partial_hof):
    partial_hof.insert(DummyIndv(0.5, 0.5))
    for indv, expected_fitness in zip(partial_hof, [0, 0.5, 1, 2]):
        assert indv.fitness == pytest.approx(expected_fitness)


def test_update_hof_with_population(all_hofs):
    population = [DummyIndv(i, i) for i in range(-7, 0)]
    population += [DummyIndv(np.nan, np.nan)]
    all_hofs.update(population)
    for i, expected_fitness in enumerate(range(-7, -2)):
        assert all_hofs[i].fitness == expected_fitness


def test_user_defined_similarity(partial_hof):
    assert len(partial_hof) == 3
    partial_hof.update([DummyIndv(1, 1.5)])
    assert len(partial_hof) == 4
    partial_hof.update([DummyIndv(1, 1)])
    assert len(partial_hof) == 4


def test_user_defined_key():
    hof = HallOfFame(5, key_function=lambda x: x.gene)
    population = [DummyIndv(-i, i) for i in range(5)]
    hof.update(population)
    assert hof[0].fitness == 0
