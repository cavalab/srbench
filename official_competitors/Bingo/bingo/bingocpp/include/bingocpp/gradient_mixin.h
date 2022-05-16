#ifndef BINGOCPP_INCLUDE_BINGOCPP_GRADIENT_MIXIN_H_
#define BINGOCPP_INCLUDE_BINGOCPP_GRADIENT_MIXIN_H_

#include <Eigen/Dense>
#include <bingocpp/equation.h>
#include <bingocpp/fitness_function.h>

#include <functional>
#include <cmath>

typedef std::tuple<double, Eigen::ArrayXd> FitnessAndGradient;
typedef std::tuple<Eigen::ArrayXd, Eigen::ArrayXXd> FitnessVectorAndJacobian;

namespace bingo {

class GradientMixin {
 public:
  virtual FitnessAndGradient GetIndividualFitnessAndGradient(Equation &individual) const = 0;
};

class VectorGradientMixin : public GradientMixin {
 public:
  VectorGradientMixin(TrainingData *training_data = nullptr, std::string metric = "mae");

  FitnessAndGradient GetIndividualFitnessAndGradient(Equation &individual) const;

  virtual FitnessVectorAndJacobian GetFitnessVectorAndJacobian(Equation &individual) const = 0;

 protected:
  static Eigen::ArrayXd mean_absolute_error_derivative(
      const Eigen::ArrayXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return (fitness_partials.rowwise() * fitness_vector.transpose().sign()).rowwise().mean();
  }

  static Eigen::ArrayXd mean_squared_error_derivative(
      const Eigen::ArrayXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 2.0 * (fitness_partials.rowwise() * fitness_vector.transpose()).rowwise().mean();
  }

  static Eigen::ArrayXd root_mean_squared_error_derivative(
      const Eigen::ArrayXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 1.0/sqrt(fitness_vector.square().mean()) *
    (fitness_partials.rowwise() * fitness_vector.transpose()).rowwise().mean();
  }

 private:
  std::function<double(Eigen::ArrayXd)> metric_function_;
  std::function<Eigen::ArrayXd(Eigen::ArrayXd, Eigen::ArrayXXd)> metric_derivative_;
};

} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_GRADIENT_MIXIN_H_
