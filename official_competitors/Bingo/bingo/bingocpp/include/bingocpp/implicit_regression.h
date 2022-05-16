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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_

#include <string>
#include <tuple>

#include <Eigen/Dense>

#include "bingocpp/equation.h"
#include "bingocpp/fitness_function.h"
#include "bingocpp/training_data.h"

#include "bingocpp/utils.h"

typedef std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> ImplicitTrainingDataState;
typedef std::tuple<ImplicitTrainingDataState, std::string, int, int> ImplicitRegressionState;


namespace bingo {

struct ImplicitTrainingData : TrainingData {
 public:
  Eigen::ArrayXXd x;

  Eigen::ArrayXXd dx_dt;

  ImplicitTrainingData(const Eigen::ArrayXXd &input) {
    InputAndDeriviative input_and_deriv = CalculatePartials(input);
    x = input_and_deriv.first;
    dx_dt = input_and_deriv.second;
  }

  ImplicitTrainingData(const Eigen::ArrayXXd &input,
                       const Eigen::ArrayXXd &derivative) {
    x = input; 
    dx_dt = derivative;
  }

  ImplicitTrainingData(ImplicitTrainingData &other) {
    x = other.x;
    dx_dt = other.dx_dt;
  }

  ImplicitTrainingData(const ImplicitTrainingDataState &state) {
    x = std::get<0>(state);
    dx_dt = std::get<1>(state);
  }

  ImplicitTrainingData* GetItem(int item);

  ImplicitTrainingData* GetItem(const std::vector<int> &items);

  ImplicitTrainingDataState DumpState() {
    return ImplicitTrainingDataState(x, dx_dt);
  }

  int Size() { 
    return x.rows();
  }
};

class ImplicitRegression : public VectorBasedFunction {
 public:
  ImplicitRegression(ImplicitTrainingData *training_data, 
                     int required_params = kNoneRequired,
                     std::string metric="mae") :
      VectorBasedFunction(new ImplicitTrainingData(*training_data), metric) {
    required_params_ = required_params;
  }

  ImplicitRegression(const ImplicitRegressionState &state):
      VectorBasedFunction(new ImplicitTrainingData(std::get<0>(state)),
                          std::get<1>(state)){
    required_params_ = std::get<2>(state);
    eval_count_ = std::get<3>(state);
  }

  ~ImplicitRegression() {
    if (training_data_ != nullptr) {
      delete training_data_;
      training_data_ = nullptr;
    }
  }

  ImplicitRegressionState DumpState();

  Eigen::ArrayXd EvaluateFitnessVector(Equation &equation) const;

 private:
  int required_params_;
  static const int kNoneRequired = -1;
};

class ImplicitRegressionSchmidt : VectorBasedFunction {

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
