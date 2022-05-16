#include <chrono>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/explicit_regression.h>
#include <bingocpp/implicit_regression.h>
#include <bingocpp/utils.h>

#include <benchmarking/benchmark_data.h>
#include <benchmarking/benchmark_logging.h>

#define EXPLICIT "explicit regression"
#define IMPLICIT "implicit regression"

using namespace bingo;

void BenchmarkRegression(std::vector<AGraph> &agraph_list,
                         const VectorBasedFunction &fitness_function);
Eigen::ArrayXd TimeBenchmark(
    void (*benchmark)(std::vector<AGraph>&, const VectorBasedFunction &),
    BenchmarkTestData &test_data,
    const VectorBasedFunction &fitness_function, int number=100, int repeat=10);
void DoRegressionBenchmarking();
void RunRegressionBenchmarks(BenchmarkTestData &benchmark_test_data);

int main() {
  DoRegressionBenchmarking();
  return 0;
}

void DoRegressionBenchmarking() {
  BenchmarkTestData benchmark_test_data;
  LoadBenchmarkData(benchmark_test_data);
  RunRegressionBenchmarks(benchmark_test_data);
}

void RunRegressionBenchmarks(BenchmarkTestData &benchmark_test_data) {
  auto input_and_derivative = CalculatePartials(benchmark_test_data.x_vals);
  auto x_vals = input_and_derivative.first;
  auto derivative = input_and_derivative.second;
  auto y = Eigen::ArrayXXd::Zero(x_vals.rows(), x_vals.cols());

  auto e_training_data = new ExplicitTrainingData(x_vals, y);
  ExplicitRegression e_regression(e_training_data);
  Eigen::ArrayXd explicit_times = TimeBenchmark(
    BenchmarkRegression, benchmark_test_data, e_regression);

  auto i_training_data = new ImplicitTrainingData(x_vals, derivative);
  ImplicitRegression i_regression(i_training_data);
  Eigen::ArrayXd implicit_times = TimeBenchmark(
    BenchmarkRegression, benchmark_test_data, i_regression);

  PrintHeader("REGRESSION BENCHMARKS");
  PrintResults(explicit_times, EXPLICIT);
  PrintResults(implicit_times, IMPLICIT);
  delete i_training_data;
  delete e_training_data;
}

Eigen::ArrayXd TimeBenchmark(
  void (*benchmark)(std::vector<AGraph>&, const VectorBasedFunction &),
  BenchmarkTestData &test_data,
  const VectorBasedFunction &fitness_function, int number, int repeat) {
  Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
  for (int run=0; run<repeat; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<number; i++) {
      benchmark(test_data.indv_list, fitness_function);	
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> time_span = (stop - start);
    times(run) = time_span.count();
  }
  return times; 
}

void BenchmarkRegression(std::vector<AGraph> &agraph_list,
                         const VectorBasedFunction &fitness_function) {
  std::vector<AGraph>::iterator indv;
  for(indv = agraph_list.begin(); indv != agraph_list.end(); indv ++) {
    fitness_function.EvaluateIndividualFitness(*indv);
  }
}