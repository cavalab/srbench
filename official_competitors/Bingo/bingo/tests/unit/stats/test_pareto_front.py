# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np
from bingo.stats.pareto_front import ParetoFront

DummyIndv = namedtuple('DummyIndv', ['fitness', 'gene', 'att1', 'att2'])


@pytest.fixture
def empty_pf():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    key_2 = lambda indv : indv.att1
    return ParetoFront(similarity_function=similar, secondary_key=key_2)


@pytest.fixture
def full_pf():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    key_2 = lambda indv : indv.att1
    hof = ParetoFront(similarity_function=similar, secondary_key=key_2)
    for i in range(5):
        hof.insert(DummyIndv(i, i, 4 - i, 4 - i))
    return hof


@pytest.fixture(params=["empty", "full"])
def all_pfs(request, empty_pf, full_pf):
    if request.param == "empty":
        return empty_pf
    return full_pf


@pytest.mark.parametrize("pop, new_len",
                        [([DummyIndv(np.nan, np.nan, 5, 3)], 5),
                         ([DummyIndv(-1, -1, 5, 3)], 6),
                         ([DummyIndv(-1, 0, 5, 3)], 5),
                         ([DummyIndv(0, -1, 5, 3)], 5),
                         ([DummyIndv(0, -1, 0, 3)], 1),
                         ([DummyIndv(0, -1, 0, 3)], 1),
                         ([DummyIndv(-1, -1, 5, 3),
                           DummyIndv(-1, -2, -1, 3)], 1),
                         ])
def test_update_adds_indvs_properly(full_pf, pop, new_len):
    full_pf.update(pop)
    assert len(full_pf) == new_len


def test_string_formatting(full_pf):
    expected_string = """0\t4\tDummyIndv(fitness=0, gene=0, att1=4, att2=4)
1\t3\tDummyIndv(fitness=1, gene=1, att1=3, att2=3)
2\t2\tDummyIndv(fitness=2, gene=2, att1=2, att2=2)
3\t1\tDummyIndv(fitness=3, gene=3, att1=1, att2=1)
4\t0\tDummyIndv(fitness=4, gene=4, att1=0, att2=0)"""
    assert str(full_pf) == expected_string


