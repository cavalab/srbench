# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import evaluation_benchmark as evaluation_benchmark
import fitness_benchmark as fitness_benchmark
import continous_local_opt_benchmarks as clo_benchmark

PROBLEM_SET_VERSION = 2


def print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    TITLE = 'USING AGRAPH PROBLEM SET # '
    TITLE += str(PROBLEM_SET_VERSION)
    NUM_STARS_LEFT_SIDE = int((80 - len(TITLE)) / 2)
    NUM_STARS_RIGHT_SIDE = int((80 - len(TITLE) + 1) / 2)

    PRINTER_LIST = [evaluation_benchmark.do_benchmarking()]
    PRINTER_LIST += fitness_benchmark.do_benchmarking()
    PRINTER_LIST += clo_benchmark.do_benchmarking(debug=False)

    print('\n\n' + '*' * NUM_STARS_LEFT_SIDE + TITLE + 
          '*' * NUM_STARS_RIGHT_SIDE)
    print("Note: Times are in milliseconds per individual\n")
    print_stats(PRINTER_LIST)

