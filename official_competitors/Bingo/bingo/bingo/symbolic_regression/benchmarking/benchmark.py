"""This module defines a Benchmark for bingo.
Benchmarks are intended to measure the performance of bingo
on common symbolic regression example problems.
"""
import numpy as np

from .. import ExplicitTrainingData


class Benchmark:
    """ The class containing the information required to run a benchmark

    Parameters
    ----------
    name : str
        Name of the benchmark
    description : str
        Description of the benchmark problem
    source : str
        Source of the benchmark problem
    training_x : numpy array
        independent variable(s) used in training. Features
    training_y : numpy array
        dependent variable used in training. Labels
    test_x : numpy array
        independent variable(s) used in testing.  Features
    test_y : numpy array
        dependent variable used in testing. Labels
    extra_info : dict (optional)
        extra information used to categorize the benchmark.  This should be a
        dictionary with string keys describing the context of the extra info.
    """
    def __init__(self, name, description, source, training_x, training_y,
                 test_x, test_y, extra_info=None):
        self.name = name
        self.description = description
        self.source = source
        self.training_data = self._make_data(training_x, training_y)
        self.test_data = self._make_data(test_x, test_y)
        self.x_dim = self.training_data.x.shape[1]
        self.extra_info = {} if extra_info is None else extra_info

    @staticmethod
    def _make_data(x, y):
        return ExplicitTrainingData(x, y)


class AnalyticBenchmark(Benchmark):
    """ The class containing the info required to run an analytic benchmark

    Parameters
    ----------
    name : str
        Name of the benchmark
    description : str
        Description of the benchmark problem
    source : str
        Source of the benchmark problem
    x_dimension : int
        The dimension of the independent variable x
    evaluation_function : callable function
        A function f that returns y = f(x)
    training_x_distribution : tuple or list(tuple)
        A tuple or list of tuples describing the distribution from which the
        training x data is drawn.  The tuples should be in the format
        ("E", a, b, c) or ("U", d, e, f). Where the first example describes
        evenly distributed data points on the interval [a, b] with increment c
        between them.  The second example describes f data points drawn from a
        random uniform distribution on the interval [d, e].
    test_x_distribution:tuple or list(tuple)
        A tuple or list of tuples describing the distribution from which the
        test x data is drawn.
    extra_info : dict (optional)
        extra information used to categorize the benchmark.  This should be a
        dictionary with string keys describing the context of the extra info.
    """

    def __init__(self, name, description, source, x_dimension,
                 evaluation_function, training_x_distribution,
                 test_x_distribution, extra_info=None):
        self._x_dim = x_dimension

        np.random.seed(0)
        training_x = self._get_x_from_distribution(training_x_distribution)
        test_x = self._get_x_from_distribution(test_x_distribution)
        np.random.seed()

        training_y = evaluation_function(training_x).reshape((-1, 1))
        test_y = evaluation_function(test_x).reshape((-1, 1))

        super().__init__(name, description, source, training_x, training_y,
                         test_x, test_y, extra_info)

    def _get_x_from_distribution(self, distribution):
        if isinstance(distribution, list):
            if len(distribution) != self._x_dim:
                raise RuntimeError("List style benchmark distributions must "
                                   "match x dimension")
            data = [self._parse_distribution(dist) for dist in distribution]
        else:
            data = [self._parse_distribution(distribution)
                    for _ in range(self._x_dim)]
        return np.hstack(data)

    @staticmethod
    def _parse_distribution(dist):
        if dist[0] == "E":
            return AnalyticBenchmark._evenly_spaced_data(dist[1], dist[2],
                                                         dist[3])
        if dist[0] == "U":
            return AnalyticBenchmark._uniform_data(dist[1], dist[2], dist[3])

        raise KeyError("benchmark distribution type {} not defined".format(
                dist[0]))

    @staticmethod
    def _evenly_spaced_data(low, high, increment):
        return np.arange(low, high+increment, increment).reshape((-1, 1))

    @staticmethod
    def _uniform_data(low, high, num_data_points):
        return np.random.uniform(low, high, num_data_points).reshape((-1, 1))
