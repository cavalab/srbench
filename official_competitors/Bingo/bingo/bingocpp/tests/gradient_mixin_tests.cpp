#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <bingocpp/gradient_mixin.h>

#include <tuple>
#include <string>
#include <vector>
#include <cmath>

using namespace bingo;

namespace {

class VectorGradFitnessFunction : public VectorGradientMixin, public VectorBasedFunction {
 public:
  VectorGradFitnessFunction(TrainingData *training_data = nullptr, std::string metric = "mae") :
      VectorGradientMixin(training_data, metric),
      VectorBasedFunction(training_data, metric) {
  }

  Eigen::ArrayXd EvaluateFitnessVector(Equation &individual) const {
    Eigen::ArrayXd fitness_vector(3);
    fitness_vector << -2.0, 0.0, 2.0;
    return fitness_vector;
  }

  FitnessVectorAndJacobian GetFitnessVectorAndJacobian(Equation &individual) const {
    Eigen::ArrayXXd jacobian(3, 2);
    jacobian << 0.5, 1.0,
                1.0, 2.0,
               -0.5, 3.0;
    return FitnessVectorAndJacobian{this->EvaluateFitnessVector(individual), jacobian};
  }
};

class ImplementedVectorMixin : public VectorGradientMixin {
 public:
  ImplementedVectorMixin(TrainingData *training_data = nullptr, std::string metric = "mae") :
      VectorGradientMixin(training_data, metric) {}

  FitnessVectorAndJacobian GetFitnessVectorAndJacobian(Equation &individual) const {
    Eigen::ArrayXd emptyVector;
    Eigen::ArrayXXd emptyArray;
    return FitnessVectorAndJacobian{emptyVector, emptyArray};
  }
};

class GradientMixinTest : public ::testing::TestWithParam<std::tuple<std::string, double, std::vector<double>>> {
 public:
  VectorGradFitnessFunction fitness_function_;
  Eigen::ArrayXd expected_gradient_;
  double expected_fitness_;

  void SetUp() {
    std::tie(fitness_function_metric_, expected_fitness_, expected_gradient_data_) = GetParam();
    expected_gradient_ = Eigen::ArrayXd(2);
    expected_gradient_ << expected_gradient_data_[0], expected_gradient_data_[1];
    fitness_function_ = VectorGradFitnessFunction(nullptr, fitness_function_metric_);
  }

  void TearDown() { }

 private:
  std::string fitness_function_metric_;
  std::vector<double> expected_gradient_data_;
};


TEST_P(GradientMixinTest, VectorIndividualFitnessAndGradient) {
  double fitness;
  Eigen::ArrayXd gradient;
  AGraph empty_individual(false);
  std::tie(fitness, gradient) = fitness_function_.GetIndividualFitnessAndGradient(empty_individual);
  ASSERT_EQ(fitness, expected_fitness_);
  ASSERT_TRUE(expected_gradient_.isApprox(gradient));
}


INSTANTIATE_TEST_SUITE_P(VectorGradientWithMetrics,
                         GradientMixinTest,
                         ::testing::Values(
                         std::make_tuple("mae", 4.0/3.0, std::vector<double> {-1.0/3.0, 2.0/3.0}),
                         std::make_tuple("mse", 8.0/3.0, std::vector<double> {-4.0/3.0, 8.0/3.0}),
                         std::make_tuple("rmse", std::sqrt(8.0/3.0), std::vector<double> {sqrt(3.0/8.0) * -2.0/3.0, sqrt(3.0/8.0) * 4.0/3.0})
                         ));

TEST(TestGradientMixin, InvalidGradientMetric) {
  try {
    ImplementedVectorMixin implemented_vector_gradient_mixin_(nullptr, "invalid_metric");
    FAIL() << "Expecting std::invalid_argument exception\n";
  } catch (std::invalid_argument &exception) {
    SUCCEED();
  }
}

} // namespace
