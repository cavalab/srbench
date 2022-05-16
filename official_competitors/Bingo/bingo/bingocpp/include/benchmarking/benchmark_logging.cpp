#include <iostream>
#include <iomanip>
#include <sstream>

#include "benchmark_logging.h"

#define EVALUATE "pure c++: evaluate"
#define X_DERIVATIVE "pure c++: x derivative"
#define C_DERIVATIVE "pure c++: c derivative"

const int kLogWidth = 78;

double StandardDeviation(const Eigen::ArrayXd &vec) {
  return std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
}

void PrintHeader(std::string title) {
  int diff = kLogWidth - title.size() - 10;
  const std::string left_tacks = std::string((diff/2), '-');
  const std::string right_tacks = std::string((diff + 1)/2, '-');
  const std::string middle = ":::: " + title + " ::::";
  const std::string full_title = left_tacks + middle + right_tacks;
  const std::string bottom = std::string (78, '-');
  std::cout << full_title << std::endl;
  OutputParams("NAME", "MEAN", "STD", " MIN", "MAX");
  std::cout << bottom << std::endl;
}

void PrintResults(const Eigen::ArrayXd &run_times, const std::string &name) {
  double std_dev = StandardDeviation(run_times);
  double average = run_times.mean();
  double max = run_times.maxCoeff();
  double min = run_times.minCoeff();
  std::string s_std_dev = StringPrecision(std_dev, 5);
  std::string s_average= StringPrecision(average, 5);
  std::string s_min= StringPrecision(min, 5);
  std::string s_max= StringPrecision(max, 5);
  OutputParams(name, s_average, s_std_dev, s_min, s_max);
}

std::string StringPrecision(double val, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << val;
  return stream.str();
}

void OutputParams(const std::string &name, const std::string &mean, 
                  const std::string &std, const std::string &min, 
                  const std::string &max) {
  std::cout << std::setw(25) << std::left << name << "   "
            << std::setw(10) << std::right << mean << " +- "
            << std::setw(10) << std::left << std << "     "
            << std::setw(10) << std::left << min << "   "
            << std::setw(10) << std::left << max 
            << std::endl;
}