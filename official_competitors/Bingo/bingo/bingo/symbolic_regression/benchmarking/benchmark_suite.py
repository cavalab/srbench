"""
This Benchmark Suite module contains a wrapper around the symbolic regression
benchmarks which are defined in the BenchmarkDefinitions Module.  The wrapper
allows for easier filtering and automatic running of benchmarks.
"""
from . import benchmark_definitions


class BenchmarkSuite:
    """Contains benchmarks used to measure performance of Bingo

    The `BenchmarkSuite` can be treated in many ways like a list in which each
    item is a `Benchmark`.  The suite can also be used to automatically run
    benchmarks using a `BenchmarkTest`

    Upon initialization the Benchmarks from the `BenchmarkDefinitions` module
    will be pulled into the the suite, including and excluding Benchmarks as
    appropriate.

    Parameters
    ----------
    inclusive_terms : list of str (optional)
        include all benchmarks in the suite that have all of these terms in
        their name
    exclusive_terms : list of str (optional)
        exclude all benchmarks in the suite that have any of these terms in
        their name

    """
    def __init__(self, inclusive_terms=None, exclusive_terms=None):
        self._benchmarks = self._find_all_benchmarks()
        if inclusive_terms is not None:
            self._filter_inclusive(inclusive_terms)
        if exclusive_terms is not None:
            self._filter_exclusive(exclusive_terms)

    @staticmethod
    def _find_all_benchmarks():
        all_benchmarks = \
            [f() for name, f in benchmark_definitions.__dict__.items()
             if callable(f) and name.startswith("bench")]
        return all_benchmarks

    def _filter_inclusive(self, terms):
        new_benchmark_list = [bench for bench in self._benchmarks
                              if BenchmarkSuite._has_terms(bench.name, terms)]
        self._benchmarks = new_benchmark_list

    @staticmethod
    def _has_terms(name, terms):
        for term in terms:
            if term not in name:
                return False
        return True

    def _filter_exclusive(self, terms):
        new_benchmark_list = [bench for bench in self._benchmarks
                              if BenchmarkSuite._hasnt_terms(bench.name,
                                                               terms)]
        self._benchmarks = new_benchmark_list

    @staticmethod
    def _hasnt_terms(name, terms):
        for term in terms:
            if term in name:
                return False
        return True

    def __len__(self):
        return len(self._benchmarks)

    def __getitem__(self, i):
        return self._benchmarks[i]

    def __iter__(self):
        return iter(self._benchmarks)

    def run_benchmark_test(self, benchmark_test, repeats=1):
        """Train and score a `BenchmarkTest` on all benchmarks in the suite

        Parameters
        ----------
        benchmark_test : BenchmarkTest
            A benchmark test that will be trained and scored
        repeats : int
            The number of repetitions to perform for each benchmark

        Returns
        -------
        training_results, testing_results : lists of list of tuple
            The training and test scores for all repeats of all benchmarks
        """
        training_results = []
        testing_results = []
        for bench in self:
            print("Running Benchmark:", bench.name)
            bench_training_results = []
            bench_testing_results = []
            for _ in range(repeats):
                benchmark_test.train(bench.training_data)
                bench_training_results.append(
                        benchmark_test.score(bench.training_data))
                bench_testing_results.append(
                        benchmark_test.score(bench.test_data))
            training_results.append(bench_training_results)
            testing_results.append(bench_testing_results)
        return training_results, testing_results

