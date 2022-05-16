"""Argument validation for functions.

This modules contains a decorator which will perform input argument validation
for a function.
"""
import functools
import logging


def _less_than_check(argument, value):
    return argument < value


def _less_than_or_equal_check(argument, value):
    return argument <= value


def _greater_than_check(argument, value):
    return argument > value


def _greater_than_or_equal_check(argument, value):
    return argument >= value


def _in_check(argument, values):
    return argument in values


def _has_check(argument, attributes):
    for att in attributes:
        if not hasattr(argument, att):
            return False
    return True


CHECKING_FUNCTIONS = {"<": _less_than_check,
                      "<=": _less_than_or_equal_check,
                      ">": _greater_than_check,
                      ">=": _greater_than_or_equal_check,
                      "in": _in_check,
                      "has": _has_check,
                      }


def argument_validation(**argchecks):
    """Function input argument validation:

    Parameters
    ----------
    argchecks : dict
        keyword arguments used as a dictionary of checks to be performed on
        input arguments

    Notes
    -----
    Defaulted arguments are ignored in checking.

    Examples
    --------
    Argument validation for x between 0-10 and y greater than 3.14


    >>> @argument_validation(x={">=": 0, "<=": 10},
    ...                      y={">": 3.14})
    ... def some_function(x, y=10):
    ...    pass

    Argument validation for algorithm  is "bubble" or "quick"

    >>> @argument_validation(algorithm={"in": ["bubble", "quick"]})
    ... def sort(algorithm):
    ...     pass

    """
    def validation_decorator(func):
        @functools.wraps(func)
        def do_validation(*pargs, **kwargs):
            check_arguments = FunctionArgChecker(func, pargs, kwargs)

            for arg_name, checks in argchecks.items():
                check_arguments(arg_name, checks)

            return func(*pargs, **kwargs)
        return do_validation
    return validation_decorator


class FunctionArgChecker:
    """A functor for managing arguments and performing validation checks

    Parameters
    ----------
    func : function
        the function for which argument checking will be performed
    positional_args : tuple
        the positional arguments passed to the function
    keyword_args : dict
        the keyword arguments passed to the function

    """
    def __init__(self, func, positional_args, keyword_args):
        self._function_name = func.__name__
        self._function_arg_names = self._get_all_argument_names(func)

        self._args = dict(keyword_args)
        self._add_positionals_to_args(positional_args)

        self._logger = logging.getLogger(func.__module__)

    @staticmethod
    def _get_all_argument_names(func):
        code = func.__code__
        return code.co_varnames[:code.co_argcount]

    def _add_positionals_to_args(self, positional_args):
        positional_names = \
            list(self._function_arg_names)[:len(positional_args)]
        for arg_name, value in zip(positional_names, positional_args):
            self._args[arg_name] = value

    def __call__(self, arg_name, checks):
        """Perform an argument validation check

        Parameters
        ----------
        arg_name : str
            name of argument to be chekced
        checks : dict {str: any}
            dictionary defining checks to perform on argument. The dictionary
            is of the form: {type_of_check: check_value, ...}
        """
        if arg_name in self._args:
            self._perform_check(arg_name, self._args[arg_name], checks)
        elif arg_name in self._function_arg_names:
            pass
        else:
            self._logger.error("Non-existent argument specified in "
                               "argument validation: %s", arg_name)
            raise SyntaxError

    def _perform_check(self, arg_name, arg_value, checks):
        for type_of_check, check_value in checks.items():
            if type_of_check not in CHECKING_FUNCTIONS:
                self._logger.error("'%s' not defined for argument validation"
                                   " of %s.",
                                   type_of_check, self._function_name)
                self._logger.error("%s:  %s %s %s", arg_name, arg_value,
                                   type_of_check, check_value)
                raise SyntaxError

            if not CHECKING_FUNCTIONS[type_of_check](arg_value, check_value):
                self._logger.error("Invalid argument in function %s.",
                                   self._function_name)
                self._logger.error("%s:  %s %s %s", arg_name, arg_value,
                                   type_of_check, check_value)
                raise ValueError
