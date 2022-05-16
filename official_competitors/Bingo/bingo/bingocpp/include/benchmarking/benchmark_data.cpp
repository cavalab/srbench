#include <fstream>
#include <istream>

#include "benchmark_data.h"

#define STACK_FILE "test-agraph-stacks.csv"
#define CONST_FILE "test-agraph-consts.csv"
#define X_FILE "test-agraph-x-vals.csv"

#define INPUT_DIM	4
#define NUM_DATA_POINTS 128 
#define STACK_SIZE 128
#define STACK_COLS 3

void LoadBenchmarkData(BenchmarkTestData &benchmark_test_data) {
  std::vector<AGraph> indv_list;
  LoadAgraphIndvidualData(indv_list);
  Eigen::ArrayXXd x_vals = LoadAgraphXVals();
  benchmark_test_data = BenchmarkTestData(indv_list, x_vals);
}

void LoadAgraphIndvidualData(std::vector<AGraph> &indv_list) {
  std::ifstream stack_filestream;
  std::ifstream const_filestream;
  stack_filestream.open(STACK_FILE);
  const_filestream.open(CONST_FILE);

  std::string stack_file_line;
  std::string const_file_line;
  while ((stack_filestream >> stack_file_line) &&
         (const_filestream >> const_file_line)) {
    AGraph curr_indv = AGraph(false);
    SetIndvStack(curr_indv, stack_file_line);
    SetIndvConstants(curr_indv, const_file_line);
    indv_list.push_back(curr_indv);
  }
  stack_filestream.close();
  const_filestream.close();
}

void SetIndvConstants(AGraph &indv, std::string &const_string) {
  std::stringstream string_stream(const_string);
  std::string num_constants;
  std::getline(string_stream, num_constants, ',');
  Eigen::VectorXd curr_const(std::stoi(num_constants));

  std::string curr_val;
  for (int i=0; std::getline(string_stream, curr_val, ','); i++) {
    curr_const(i) = std::stod(curr_val);
  }
  indv.SetLocalOptimizationParams(curr_const);
}

void SetIndvStack(AGraph &indv, std::string &stack_string) {
  std::stringstream string_stream(stack_string);
  Eigen::ArrayX3i curr_stack = Eigen::ArrayX3i(STACK_SIZE, STACK_COLS);

  std::string curr_op;
  for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
    curr_stack(i/STACK_COLS, i%STACK_COLS) = std::stoi(curr_op);
  }
  indv.SetCommandArray(curr_stack);
}

Eigen::ArrayXXd LoadAgraphXVals() {
  std::ifstream filename;
  filename.open(X_FILE);

  Eigen::ArrayXXd x_vals = Eigen::ArrayXXd(NUM_DATA_POINTS, INPUT_DIM);
  std::string curr_x_row;
  for (int row = 0; filename >> curr_x_row; row++) {
    std::stringstream string_stream(curr_x_row);
    std::string curr_x;
    for (int col = 0; std::getline(string_stream, curr_x, ','); col++) {
      x_vals(row, col) = std::stod(curr_x);
    }
  }
  filename.close();
  return x_vals;
}