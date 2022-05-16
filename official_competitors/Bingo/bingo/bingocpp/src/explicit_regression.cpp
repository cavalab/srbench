#include <iostream>
#include <tuple>

#include "bingocpp/explicit_regression.h"

namespace bingo {

ExplicitTrainingData *ExplicitTrainingData::GetItem(int item) {
  return new ExplicitTrainingData(x.row(item), y.row(item));
}

ExplicitTrainingData *ExplicitTrainingData::GetItem(
    const std::vector<int> &items) {
  Eigen::ArrayXXd temp_in(items.size(), x.cols());
  Eigen::ArrayXXd temp_out(items.size(), y.cols());

  for (unsigned int row = 0; row < items.size(); row ++) {
    temp_in.row(row) = x.row(items[row]);
    temp_out.row(row) = y.row(items[row]);
  }

  return new ExplicitTrainingData(temp_in, temp_out);
}

Eigen::ArrayXd ExplicitRegression::EvaluateFitnessVector(
    Equation &individual) const {
  ++ eval_count_;
  const Eigen::ArrayXXd x = ((ExplicitTrainingData*)training_data_)->x;
  Eigen::ArrayXXd f_of_x = individual.EvaluateEquationAt(x);
  Eigen::ArrayXXd error = f_of_x - ((ExplicitTrainingData*)training_data_)->y;
  if (relative_)
    error /= ((ExplicitTrainingData*)training_data_)->y;
  return error;
}

FitnessVectorAndJacobian ExplicitRegression::GetFitnessVectorAndJacobian(
    Equation &individual) const {
  ++ eval_count_;
  Eigen::ArrayXXd f_of_x, df_dc;
  const Eigen::ArrayXXd x = ((ExplicitTrainingData*)training_data_)->x;
  std::tie(f_of_x, df_dc) = individual.EvaluateEquationWithLocalOptGradientAt(x);

  Eigen::ArrayXXd error = f_of_x - ((ExplicitTrainingData*)training_data_)->y;
  if (relative_) {
    error /= ((ExplicitTrainingData*)training_data_)->y;
    df_dc.colwise() /= ((ExplicitTrainingData*)training_data_)->y(Eigen::all, 0);
  }
  return FitnessVectorAndJacobian{error, df_dc};
}

ExplicitRegressionState ExplicitRegression::DumpState() {
  return ExplicitRegressionState(
          ((ExplicitTrainingData*)training_data_)->DumpState(),
          metric_, eval_count_);
}

} // namespace bingo