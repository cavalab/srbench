#include <chrono>

#include <benchmarking/benchmark_data.h>
#include <benchmarking/benchmark_logging.h>

#define EVALUATE "pure c++: evaluate"
#define X_DERIVATIVE "pure c++: x derivative"
#define C_DERIVATIVE "pure c++: c derivative"

void DoBenchmarking();
Eigen::ArrayXd TimeBenchmark(
  void (*benchmark)(const std::vector<AGraph>&, const Eigen::ArrayXXd&), 
  const BenchmarkTestData &test_data, int number=100, int repeat=10);
void RunBenchmarks(const BenchmarkTestData &benchmark_test_data);
void BenchmarkEvaluate(const std::vector<AGraph> &indv_list,
                       const Eigen::ArrayXXd &x_vals);
void BenchmarkEvaluateAndXDerivative(const std::vector<AGraph> &indv_list,
                                     const Eigen::ArrayXXd &x_vals);
void BenchmarkEvaluateAndCDerivative(const std::vector<AGraph> &indv_list,
                                     const Eigen::ArrayXXd &x_vals);

int main() {
  DoBenchmarking();
  return 0;
}

void DoBenchmarking() {
  BenchmarkTestData benchmark_test_data =  BenchmarkTestData();
  LoadBenchmarkData(benchmark_test_data);
  RunBenchmarks(benchmark_test_data);
}

void RunBenchmarks(const BenchmarkTestData &benchmark_test_data) {
  Eigen::ArrayXd evaluate_times = TimeBenchmark(BenchmarkEvaluate, benchmark_test_data);
  Eigen::ArrayXd x_derivative_times = TimeBenchmark(BenchmarkEvaluateAndXDerivative, benchmark_test_data);
  Eigen::ArrayXd c_derivative_times = TimeBenchmark(BenchmarkEvaluateAndCDerivative, benchmark_test_data);
  PrintHeader();
  PrintResults(evaluate_times, EVALUATE);
  PrintResults(x_derivative_times, X_DERIVATIVE);
  PrintResults(c_derivative_times, C_DERIVATIVE);
}

Eigen::ArrayXd TimeBenchmark(
  void (*benchmark)(const std::vector<AGraph>&, const Eigen::ArrayXXd&), 
  const BenchmarkTestData &test_data, int number, int repeat) {
  Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
  for (int run=0; run<repeat; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<number; i++) {
      benchmark(test_data.indv_list, test_data.x_vals);	
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> time_span = (stop - start);
    times(run) = time_span.count();
  }
  return times; 
}

void BenchmarkEvaluate(const std::vector<AGraph> &indv_list,
                       const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    evaluation_backend::Evaluate(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams());
  } 
}

void BenchmarkEvaluateAndXDerivative(const std::vector<AGraph> &indv_list,
                                     const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    evaluation_backend::EvaluateWithDerivative(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams(), true);
  }
}

void BenchmarkEvaluateAndCDerivative(const std::vector<AGraph> &indv_list,
                                     const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    evaluation_backend::EvaluateWithDerivative(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams(), false);
  }
}