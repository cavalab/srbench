# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.util.probability_mass_function import ProbabilityMassFunction


@pytest.fixture(params=[True, False])
def constant_pmf(request):
    if request.param:
        return (ProbabilityMassFunction(items=[True, False], weights=[1, 0]),
                request.param)
    return (ProbabilityMassFunction(items=[True, False], weights=[0, 1]),
            request.param)


@pytest.fixture
def empty_pmf():
    return ProbabilityMassFunction()


@pytest.fixture
def sample_pmf():
    return ProbabilityMassFunction(items=[1.0, 2.0, 3.0, 4.0],
                                   weights=[4.0, 3.0, 2.0, 1.0])


@pytest.fixture
def equal_pmf():
    return ProbabilityMassFunction(items=[True, False])


def test_raises_exception_for_uneven_init():
    with pytest.raises(ValueError):
        ProbabilityMassFunction(items=[1, 2, 3], weights=[1, ])


def test_raises_exception_for_non_listlike_init():
    with pytest.raises(AttributeError):
        pmf = ProbabilityMassFunction(items="happy")
        pmf.add_item("sad")


def test_raises_exception_for_non_numeric_weight():
    with pytest.raises(TypeError):
        ProbabilityMassFunction(items=[1, 2, 3], weights=[1, "a", 3])


def test_raises_exception_for_draw_from_empty_pmf(empty_pmf):
    with pytest.raises(IndexError):
        _ = empty_pmf.draw_sample()


def test_constant_pmfs(constant_pmf):
    pmf, expected_value = constant_pmf
    for _ in range(10):
        assert pmf.draw_sample() == expected_value


@pytest.mark.parametrize("item", [True, "ABC", sum])
@pytest.mark.parametrize("weight", [None, 1.0])
def test_add_items_to_empty_pmf(empty_pmf, item, weight):
    empty_pmf.add_item(item, weight)
    assert empty_pmf.draw_sample() == item


@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
@pytest.mark.filterwarnings("ignore:invalid value encountered in reduce")
def test_raises_exception_negative_weights(empty_pmf):
    empty_pmf.add_item("-", -1.0)
    with pytest.raises(ValueError):
        empty_pmf.add_item("+", 1.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
def test_raises_exception_nan_weights(empty_pmf):
    with pytest.raises(ValueError):
        empty_pmf.add_item("a", 0.0)


def test_default_equal_weight_init(equal_pmf):
    assert equal_pmf.normalized_weights[0] == equal_pmf.normalized_weights[1]


def test_default_equal_weight_added(equal_pmf):
    equal_pmf.add_item(True)

    assert equal_pmf.normalized_weights[2] == equal_pmf.normalized_weights[0]
    assert equal_pmf.normalized_weights[2] == equal_pmf.normalized_weights[1]
