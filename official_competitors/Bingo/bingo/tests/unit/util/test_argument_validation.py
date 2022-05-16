# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.util.argument_validation import argument_validation


@pytest.mark.parametrize("func_arg,check", [
    (-0.1, {">=": 0}),
    (-0.1, {">": 0}),
    (0, {">": 0.0}),
    (0.1, {"<=": 0}),
    (0.1, {"<": 0}),
    (0, {"<": 0}),
    ("A", {"in": [1, 2]}),
    ("A", {"in": "BCDE"}),
    ("A", {"in": ["ABCDE", "BCD"]}),
    ("A", {"has": ["istitle", "__call__"]})
])
@pytest.mark.parametrize("keyword", [True, False])
def test_raises_error_failed_check(func_arg, check, keyword):
    @argument_validation(arg=check)
    def test(arg):
        print(arg)
    with pytest.raises(ValueError):
        if keyword:
            test(arg=func_arg)
        else:
            test(func_arg)


@pytest.mark.parametrize("check_type", [">=", ">", "<=", "<"])
@pytest.mark.parametrize("keyword", [True, False])
def test_raises_error_wrong_type_for_check(check_type, keyword):
    @argument_validation(arg={check_type: 0})
    def test(arg):
        print(arg)
    with pytest.raises(TypeError):
        if keyword:
            test(arg="string")
        else:
            test("string")


@pytest.mark.parametrize("func_arg,check", [
    (0.1, {">=": 0}),
    (0.1, {">": 0}),
    (0, {">=": 0.0}),
    (-0.1, {"<=": 0}),
    (-0.1, {"<": 0}),
    (0, {"<=": 0}),
    ("A", {"in": [1, "A"]}),
    ("A", {"in": "ABCDE"}),
    ("A", {"has": ["istitle"]}),
])
@pytest.mark.parametrize("keyword", [True, False])
def test_valid_checks(func_arg, check, keyword):
    @argument_validation(arg=check)
    def test(arg):
        print(arg)
    if keyword:
        test(arg=func_arg)
    else:
        test(func_arg)


@pytest.mark.parametrize("default,check", [
    (-0.1, {">=": 0}),
    ("A", {"in": [1, 2]}),
    ("A", {">": 1}),
])
def test_ignoring_defaults(default, check):
    @argument_validation(arg=check)
    def test(arg=default):
        print(arg)
    test()


def test_raises_error_nonexisting_argument():
    @argument_validation(arg2={"<=": 0})
    def test(arg):
        print(arg)
    with pytest.raises(SyntaxError):
        test(-1)


def test_raises_error_nonexisting_check():
    @argument_validation(arg={"==": 0})
    def test(arg):
        print(arg)
    with pytest.raises(SyntaxError):
        test(0)
