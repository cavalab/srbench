/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
#ifndef BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <functional>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/equation.h>
#include <bingocpp/training_data.h>

namespace metric_functions {

const std::unordered_set<std::string> kMeanAbsoluteError = {
  "mean absolute error",
  "mae"
};

const std::unordered_set<std::string> kMeanSquaredError = {
  "mean squared error",
  "mse"
};

const std::unordered_set<std::string> kRootMeanSquaredError = {
  "root mean squared error",
  "rmse"
};

inline bool metric_found(const std::unordered_set<std::string> &set,
                         std::string metric) {
  return set.find(metric) != set.end();
}

inline double mean_absolute_error(const Eigen::ArrayXd &fitness_vector) {
  return fitness_vector.abs().mean();
}

inline double root_mean_squared_error(const Eigen::ArrayXd &fitness_vector) {
  return sqrt(fitness_vector.square().mean());
}

inline double mean_squared_error(const Eigen::ArrayXd &fitness_vector) {
  return fitness_vector.square().mean();
}
} // namespace metric_functions

namespace bingo {

class FitnessFunction {
 public:
  inline FitnessFunction(TrainingData *training_data = nullptr) :
    eval_count_(0), training_data_(training_data) { }

  virtual ~FitnessFunction() { }

  virtual double EvaluateIndividualFitness(Equation &individual) const = 0;

  int GetEvalCount() const {
    return eval_count_;
  }

  void SetEvalCount(int eval_count) {
    eval_count_ = eval_count;
  }

  TrainingData* GetTrainingData() const {
    return training_data_;
  }

  void SetTrainingData(TrainingData* training_data) {
    training_data_ = training_data;
  }

 protected:
  mutable int eval_count_;
  TrainingData* training_data_;
};

class VectorBasedFunction : public FitnessFunction {
 public:
  typedef double (VectorBasedFunction::* MetricFunctionPointer)(const Eigen::ArrayXXd&);
  VectorBasedFunction(TrainingData *training_data = nullptr,
                      std::string metric = "mae") :
      FitnessFunction(training_data), metric_(metric) {
    metric_function_ = GetMetric(metric);
  }

  virtual ~VectorBasedFunction() { }

  double EvaluateIndividualFitness(Equation &individual) const {
    Eigen::ArrayXd fitness_vector = EvaluateFitnessVector(individual);
    return this->metric_function_(fitness_vector);
  }

  virtual Eigen::ArrayXd
  EvaluateFitnessVector(Equation &individual) const = 0;

 protected:
  std::string metric_;

  std::function<double(Eigen::ArrayXd)> GetMetric(std::string metric) {
    if (metric_functions::metric_found(metric_functions::kMeanAbsoluteError, metric)) {
      return metric_functions::mean_absolute_error;
    } else if (metric_functions::metric_found(metric_functions::kMeanSquaredError, metric)) {
      return metric_functions::mean_squared_error;
    } else if (metric_functions::metric_found(metric_functions::kRootMeanSquaredError, metric)) {
      return metric_functions::root_mean_squared_error;
    } else {
      throw std::invalid_argument("Invalid metric for Fitness Function");
    }
  }

 private:
  std::function<double(Eigen::ArrayXd)> metric_function_;
};
} // namespace bingo

#endif // BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_