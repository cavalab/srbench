"""
Definitions of Benchmarks that can be used a `BenchmarkingSuite`

Benchmarks are created in this module by defining a function with a name
starting with 'bench_` that takes no input parameters and returns a `Benchmark`

Each benchmark includes its source from the literature, but most of this
collection was taken from the ones suggested by McDermott et al. (2012)
"""
# pylint: disable=missing-docstring
import numpy as np
from .benchmark import AnalyticBenchmark


def bench_koza_1():
    name = "Koza-1"
    description = "The polynomial x^4 + x^3 + x^2 + x"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**4 + x**3 + x**2 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_koza_2():
    name = "Koza-2"
    description = "The polynomial x^5 - 2x^3 + x"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**5 - 2*x**3 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_koza_3():
    name = "Koza-3"
    description = "The polynomial x^6 - 2x^4 + x^2"
    source = "J.R. Koza. Genetic Programming: On the Programming of " + \
             "Computers by Means of Natural selection. MIT Press 1992"
    x_dim = 1

    def eval_func(x):
        return x**6 - 2*x**4 + x**2

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_1():
    name = "Nguyen-1"
    description = "The polynomial x^3 + x^2 + x"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return x**3 + x**2 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_3():
    name = "Nguyen-3"
    description = "The polynomial x^5 + x^4 + x^3 + x^2 + x"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return x**5 + x**4 + x**3 + x**2 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_4():
    name = "Nguyen-4"
    description = "The polynomial x^6 + x^5 + x^4 + x^3 + x^2 + x"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return x**6 + x**5 + x**4 + x**3 + x**2 + x

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_5():
    name = "Nguyen-5"
    description = "sin(x^2)cos(x) - 1"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return np.sin(x**2) * np.cos(x) - 1

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_6():
    name = "Nguyen-6"
    description = "sin(x) + sin(x + x^2)"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return np.sin(x) + np.sin(x + x**2)

    train_dist = ("U", -1, 1, 20)
    test_dist = ("U", -1, 1, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_7():
    name = "Nguyen-7"
    description = "ln(x + 1) + ln(x^2 + 1)"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return np.log(x + 1) + np.log(x**2 + 1)

    train_dist = ("U", 0, 2, 20)
    test_dist = ("U", 0, 2, 20)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_8():
    name = "Nguyen-8"
    description = "sqrt(x)"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 1

    def eval_func(x):
        return np.sqrt(x)

    train_dist = ("U", 0, 4, 20)
    test_dist = ("U", 0, 4, 20)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_9():
    name = "Nguyen-9"
    description = "sin(x_0) + sin(x_1**2)"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 2

    def eval_func(x):
        return np.sin(x[:, 0]) + np.sin(x[:, 1]**2)

    train_dist = ("U", -1, 1, 100)
    test_dist = ("U", -1, 1, 100)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_nguyen_10():
    name = "Nguyen-10"
    description = "2sin(x_0)cos(x_1)"
    source = "Q.U. Nguyen, et al. Symantically-Based Crossover in Genetic " + \
             "Programming: Application to Real-valued Symbolic Regression." + \
             "GPEM 2011"
    x_dim = 2

    def eval_func(x):
        return 2 * np.sin(x[:, 0]) * np.cos(x[:, 1])

    train_dist = ("U", -1, 1, 100)
    test_dist = ("U", -1, 1, 100)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_paige_1():
    name = "Paige-1"
    description = "1/(1+x_0^(-4)) + 1/(1+x_1^(-4))"
    source = "L. Pagie and P. Hogeweg. Evolutionary Consequences of " + \
             "Coevolving Targets. Evolutionary Computation, 1997."
    x_dim = 2

    def eval_func(x):
        return 1/(1+x[:, 0]**-4) + 1/(1+x[:, 1]**-4)

    train_dist = ("E", -5, 5, 0.4)
    test_dist = ("U", -5, 5, 25)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_1():
    name = "Korns-1"
    description = "1.57 + 24.3(x_0)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 1.57 + 24.3 * x[:, 0]

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_2():
    name = "Korns-2"
    description = "0.23 + 14.2 * (x_0 - x_2)/(3 * x_1)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 0.23 + 14.2 * (x[:, 0] - x[:, 2])/(3 * x[:, 1])

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_3():
    name = "Korns-3"
    description = "-5.41 + 4.9 * (x_0 - x_2 + x_3 / x_1) / (3 * x_1)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return -5.41 + 4.9 * (x[:, 0] - x[:, 2] + x[:, 3] / x[:, 1]) / \
               (3 * x[:, 1])

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_4():
    name = "Korns-4"
    description = "-2.3 + 0.13 * sin(x_4)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return -2.3 + 0.13 * np.sin(x[:, 4])

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_5():
    name = "Korns-5"
    description = "3 + 2.13 * ln(|x_1|)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 3 + 2.13 * np.log(np.abs(x[:, 1]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_6():
    name = "Korns-6"
    description = "1.3 + 0.13 * sqrt(|x_2|)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 1.3 + 0.13 * np.sqrt(np.abs(x[:, 2]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_7():
    name = "Korns-7"
    description = "213.80940899 * (1 - exp(-0.54723748542 * x_2))"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 213.80940899 * (1 - np.exp(-0.54723748542 * x[:, 2]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_8():
    name = "Korns-8"
    description = "6.87 + 11 * sqrt(|7.23 * x_0 * x_1 * x_2|)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 6.87 + 11 * np.sqrt(np.abs(7.23 * x[:, 0] * x[:, 1] * x[:, 2]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_9():
    name = "Korns-9"
    description = "(sqrt(|x_2|) * exp(x_1)) / (ln(|x_3|) * x_0^2)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return (np.sqrt(np.abs(x[:, 2])) * np.exp(x[:, 1])) / \
               (np.log(np.abs(x[:, 3])) * x[:, 0]**2)

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_10():
    name = "Korns-10"
    description = "0.81 + 24.3 * (2 * x_3 + 3 * x_4^2) / " + \
                  "(4 * x_0^3 + 5 * x_1^4)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 0.81 + 24.3 * (2 * x[:, 3] + 3 * x[:, 4]**2) / \
               (4*x[:, 0]**3 + 5 * x[:, 1]**4)

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_11():
    name = "Korns-11"
    description = "6.87 + 11 * cos(7.23 * x_2^3)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 6.87 + 11 * np.cos(7.23 * x[:, 2]**3)

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_12():
    name = "Korns-12"
    description = "2 - 2.1 * np.cos(9.8 * x_2) * np.sin(1.3 * x_1)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 2 - 2.1 * np.cos(9.8 * x[:, 2]) * np.sin(1.3 * x[:, 1])

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 4}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_13():
    name = "Korns-13"
    description = "32 - 3 * (tan(x_2) * tan(x_4)) / (tan(x_3) * tan(x_0))"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 32 - 3 * (np.tan(x[:, 2]) * np.tan(x[:, 4])) / \
               (np.tan(x[:, 3]) * np.tan(x[:, 0]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_14():
    name = "Korns-14"
    description = "22 - 4.2*(cos(x_2) - tan(x_3)) * tanh(x_4) / sin(x_0)"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 22 - 4.2*(np.cos(x[:, 2]) - np.tan(x[:, 3])) * \
               np.tanh(x[:, 4]) / np.sin(x[:, 0])

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_korns_15():
    name = "Korns-15"
    description = "12 - 6 * tan(x_2) / exp(x_3) * (log(|x_4|) - tan(x_0))"
    source = "M. F. Korns. Accuracy in Symbolic Regression. In Proc. GPTP. " +\
             "2011."
    x_dim = 5

    def eval_func(x):
        return 12 - 6 * np.tan(x[:, 2]) / np.exp(x[:, 3]) * \
               (np.log(np.abs(x[:, 4])) - np.tan(x[:, 0]))

    train_dist = ("U", -50, 50, 10000)
    test_dist = ("U", -50, 50, 10000)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_1():
    name = "Keijzer-1"
    description = "0.3 * x * sin(2pi * x)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return 0.3 * x * np.sin(2 * np.pi * x)

    train_dist = ("E", -1, 1, 0.1)
    test_dist = ("E", -1, 1, 0.001)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_2():
    name = "Keijzer-2"
    description = "0.3 * x * sin(2pi * x)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return 0.3 * x * np.sin(2 * np.pi * x)

    train_dist = ("E", -2, 2, 0.1)
    test_dist = ("E", -2, 2, 0.001)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_3():
    name = "Keijzer-3"
    description = "0.3 * x * sin(2pi * x)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return 0.3 * x * np.sin(2 * np.pi * x)

    train_dist = ("E", -3, 3, 0.1)
    test_dist = ("E", -3, 3, 0.001)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_4():
    name = "Keijzer-4"
    description = "x^3 * exp(-x) * cos(x) * sin(x) * (sin(x)^2 * cos(x) - 1)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return x**3 * np.exp(-x) * np.cos(x) * np.sin(x) * \
               (np.sin(x)**2 * np.cos(x) - 1)

    train_dist = ("E", 0, 10, 0.05)
    test_dist = ("E", 0.05, 10.05, 0.05)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_5():
    name = "Keijzer-5"
    description = "30 * x_0 * x_2 / ((x_0 - 10) * x_1^2)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 3

    def eval_func(x):
        return 30 * x[:, 0] * x[:, 2] / ((x[:, 0] - 10) * x[:, 1]**2)

    train_dist = [("U", -1, 1, 1000), ("U", 1, 2, 1000), ("U", -1, 1, 1000)]
    test_dist = [("U", -1, 1, 1000), ("U", 1, 2, 1000), ("U", -1, 1, 1000)]
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_6():
    name = "Keijzer-6"
    description = "sum from 1 to x of 1/x_i"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i:] += 1/x[i:]
        return y

    train_dist = ("E", 1, 50, 1)
    test_dist = ("E", 1, 120, 1)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_7():
    name = "Keijzer-7"
    description = "ln(x)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return np.log(x)

    train_dist = ("E", 1, 100, 1)
    test_dist = ("E", 1, 100, 0.1)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_8():
    name = "Keijzer-8"
    description = "sqrt(x)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return np.sqrt(x)

    train_dist = ("E", 0, 100, 1)
    test_dist = ("E", 0, 100, 0.1)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_9():
    name = "Keijzer-9"
    description = "arcsinh(x), i.e. ln(x + sqrt(x^2 + 1))"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 1

    def eval_func(x):
        return np.arcsinh(x)

    train_dist = ("E", 0, 100, 1)
    test_dist = ("E", 0, 100, 0.1)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_10():
    name = "Keijzer-10"
    description = "x_0^(x_1)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return np.power(x[:, 0], x[:, 1])

    train_dist = ("U", 0, 1, 100)
    test_dist = ("E", 0, 1, 0.01)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_11():
    name = "Keijzer-11"
    description = "x_0 * x_1 + sin((x_0 - 1) * (x_1 - 1))"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return x[:, 0] * x[:, 1] + np.sin((x[:, 0] - 1) * (x[:, 1] - 1))

    train_dist = ("U", -3, 3, 20)
    test_dist = ("E", -3, 3, 0.01)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_12():
    name = "Keijzer-12"
    description = "x_0^4 - x_0^3 + 0.5 * x_1^2 - x_1"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return x[:, 0]**4 - x[:, 0]**3 + 0.5 * x[:, 1]**2 - x[:, 1]

    train_dist = ("U", -3, 3, 20)
    test_dist = ("E", -3, 3, 0.01)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_13():
    name = "Keijzer-13"
    description = "6 * sin(x_0) * cos(x_1)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return 6 * np.sin(x[:, 0]) * np.cos(x[:, 1])

    train_dist = ("U", -3, 3, 20)
    test_dist = ("E", -3, 3, 0.01)
    extra_info = {"const_dim": 0}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_14():
    name = "Keijzer-14"
    description = "8 / (2 + x_0^2 + x_1^2)"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return 8 / (2 + x[:, 0]**2 + x[:, 1]**2)

    train_dist = ("U", -3, 3, 20)
    test_dist = ("E", -3, 3, 0.01)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_keijzer_15():
    name = "Keijzer-15"
    description = "x_0^3 / 5 + x_1^3 / 2 - x_1 - x_0"
    source = "M. Keijzer. Improving Symbolic Regression with Interval " + \
             "Arithmetic and Linear Scaling. In Proc. EuroGP. 2003."
    x_dim = 2

    def eval_func(x):
        return x[:, 0]**3 / 5 + x[:, 1]**3 / 2 - x[:, 1] - x[:, 0]

    train_dist = ("U", -3, 3, 20)
    test_dist = ("E", -3, 3, 0.01)
    extra_info = {"const_dim": 2}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_1():
    name = "Vladislavleva-1"
    description = "exp(-(x_0 - 1)^2) / (1.2 + (x_1 - 2.5)^2)"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 2

    def eval_func(x):
        return np.exp(-(x[:, 0] - 1)**2) / (1.2 + (x[:, 1] - 2.5)**2)

    train_dist = ("U", 0.3, 4, 100)
    test_dist = ("E", -0.2, 4.2, 0.1)
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_2():
    name = "Vladislavleva-2"
    description = "exp(-x) * x^3 * cos(x) * sin(x) * (cos(x) * sin(x)^2 - 1)"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 1

    def eval_func(x):
        return np.exp(-x[:, 0]) * x[:, 0]**3 * np.cos(x[:, 0]) * \
               np.sin(x[:, 0]) * (np.cos(x[:, 0]) * np.sin(x[:, 0])**2 - 1)

    train_dist = ("E", 0.05, 10, 0.1)
    test_dist = ("E", -0.5, 10.5, 0.05)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_3():
    name = "Vladislavleva-3"
    description = "exp(-x_0) * x_0^3 * cos(x_0) * sin(x_0) * " + \
                  "(cos(x_0) * sin(x_0)^2 - 1) * (x_1 - 5) " + \
                  " Note that the training/testing inputs differ slightly " + \
                  "from the ones described in McDermot et al."
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 2

    def eval_func(x):
        return np.exp(-x[:, 0]) * x[:, 0]**3 * np.cos(x[:, 0]) * \
               np.sin(x[:, 0]) * (np.cos(x[:, 0]) * np.sin(x[:, 0])**2 - 1) * \
               (x[:, 1] - 5)

    train_dist = [("E", 0.05, 10, 0.1), ("U", 0.05, 10.05, 101)]
    test_dist = [("E", -0.5, 10.5, 0.05), ("U", 0.05, 10.05, 221)]
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_4():
    name = "Vladislavleva-4"
    description = "10 / (5 + (x_0 - 3)^2 + (x_1 - 3)^2 + (x_2 - 3)^2 + " + \
                  "(x_3 - 3)^2 + (x_4 - 3)^2)"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 5

    def eval_func(x):
        return 10 / (5 + (x[:, 0] - 3)**2 + (x[:, 1] - 3)**2 +
                     (x[:, 2] - 3)**2 + (x[:, 3] - 3)**2 + (x[:, 4] - 3)**2)

    train_dist = ("U", 0.05, 6.05, 1024)
    test_dist = ("U", -0.25, 6.35, 5000)
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_5():
    name = "Vladislavleva-5"
    description = "30 * (x_0 - 1) * (x_2 - 1) / (x_1^2 * (x_0 - 10))" + \
                  " Note that the testing inputs differ slightly " + \
                  "from the ones described in McDermot et al."
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 3

    def eval_func(x):
        return 30 * (x[:, 0] - 1) * (x[:, 2] - 1) / \
               (x[:, 1]**2 * (x[:, 0] - 10))

    train_dist = [("U", 0.05, 2, 300), ("U", 1, 2, 300), ("U", 0.05, 2, 300)]
    test_dist = [("U", -0.05, 2.1, 2156), ("U", 0.95, 2.05, 2156),
                 ("U", -0.05, 2.1, 2156)]
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_6():
    name = "Vladislavleva-6"
    description = "6 * sin(x_0]) * cos(x_1)"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 2

    def eval_func(x):
        return 6 * np.sin(x[:, 0]) * np.cos(x[:, 1])

    train_dist = ("U", 0.1, 5.9, 30)
    test_dist = ("E", -0.05, 6.05, .02)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_7():
    name = "Vladislavleva-7"
    description = "(x_0 - 3) * (x_1 - 3) + 2 * sin((x_0 - 4) * (x_1 - 4))"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 2

    def eval_func(x):
        return (x[:, 0] - 3) * (x[:, 1] - 3) + \
               2 * np.sin((x[:, 0] - 4) * (x[:, 1] - 4))

    train_dist = ("U", 0.05, 6.05, 300)
    test_dist = ("U", -0.25, 6.35, 1000)
    extra_info = {"const_dim": 1}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)


def bench_vladislavleva_8():
    name = "Vladislavleva-8"
    description = "(x_0 - 3) * (x_1 - 3) + 2 * sin((x_0 - 4) * (x_1 - 4))"
    source = "E. Vladislavleva, G. Smits, and D. Den Hertog. Order of " + \
             "Nonlinearity as a Complexity Measure for Models Generated by" + \
             " Symbolic Regression via Pareto Genetic Programming. IEEE " + \
             "Trans EC, 13(2):333–349, 2009."
    x_dim = 2

    def eval_func(x):
        return ((x[:, 0] - 3)**4 + (x[:, 1] - 3)**4 - (x[:, 1] - 3)) / \
               ((x[:, 1] - 2)**4 + 10)

    train_dist = ("U", 0.05, 6.05, 50)
    test_dist = ("E", -0.25, 6.35, 0.2)
    extra_info = {"const_dim": 3}
    return AnalyticBenchmark(name, description, source, x_dim, eval_func,
                             train_dist, test_dist, extra_info)
