#include <Eigen/Dense>
#include <bingocpp/gradient_mixin.h>

#include <tuple>

namespace bingo {

VectorGradientMixin::VectorGradientMixin(TrainingData *training_data, std::string metric) {
  if (metric_functions::metric_found(metric_functions::kMeanAbsoluteError, metric)) {
    metric_function_ = metric_functions::mean_absolute_error;
    metric_derivative_ = VectorGradientMixin::mean_absolute_error_derivative;
  } else if (metric_functions::metric_found(metric_functions::kMeanSquaredError, metric)) {
    metric_function_ = metric_functions::mean_squared_error;
    metric_derivative_ = VectorGradientMixin::mean_squared_error_derivative;
  } else if (metric_functions::metric_found(metric_functions::kRootMeanSquaredError, metric)) {
    metric_function_ = metric_functions::root_mean_squared_error;
    metric_derivative_ = VectorGradientMixin::root_mean_squared_error_derivative;
  } else {
    throw std::invalid_argument("Invalid metric for VectorGradientMixin");
  }
}

FitnessAndGradient VectorGradientMixin::GetIndividualFitnessAndGradient(Equation &individual) const {
  Eigen::ArrayXd fitness_vector;
  Eigen::ArrayXXd jacobian;
  std::tie(fitness_vector, jacobian) = this->GetFitnessVectorAndJacobian(individual);
  double fitness = this->metric_function_(fitness_vector);
  return FitnessAndGradient{fitness, metric_derivative_(fitness_vector, jacobian.transpose())};
}

} // namespace bingo
