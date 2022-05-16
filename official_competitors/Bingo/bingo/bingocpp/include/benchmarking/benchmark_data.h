#ifndef APP_BENCMARK_UTILS_BENCHMARK_DATA_H_
#define APP_BENCMARK_UTILS_BENCHMARK_DATA_H_

#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/agraph.h>

using namespace bingo;

struct BenchmarkTestData {
  std::vector<AGraph> indv_list;
  Eigen::ArrayXXd x_vals;
  BenchmarkTestData() {}
  BenchmarkTestData(std::vector<AGraph> &il, Eigen::ArrayXXd &x):
    indv_list(il), x_vals(x) {}
};

void LoadBenchmarkData(BenchmarkTestData &benchmark_test_data);
void LoadAgraphIndvidualData(std::vector<AGraph> &indv_list);
void SetIndvConstants(AGraph &indv, std::string &const_string);
void SetIndvStack(AGraph &indv, std::string &stack_string);
Eigen::ArrayXXd LoadAgraphXVals();
double StandardDeviation(const Eigen::ArrayXd &vec);

#endif // APP_BENCMARK_UTILS_BENCHMARK_DATA_H_
