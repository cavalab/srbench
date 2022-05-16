#include <cmath>
#include <tuple>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <bingocpp/explicit_regression.h>

#include "test_fixtures.h"
#include "testing_utils.h"

using namespace bingo;

namespace {

class TestExplicitRegression : public testing::Test {
 public:
  ExplicitTrainingData* training_data_;
  testutils::SumEquation sum_equation_;

  void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  void TearDown() {
    delete training_data_;
  }

 private:
  ExplicitTrainingData* init_sample_training_data() {
    Eigen::ArrayXXd x = Eigen::ArrayXXd::Constant(10, 5, 1.0);
    Eigen::Array<double, 10, 1> y = Eigen::ArrayXd::Constant(10, 1, 2.5);
    return new ExplicitTrainingData(x, y);
  }
};

TEST_F(TestExplicitRegression, EvaluateIndividualFitness) {
  ExplicitRegression regressor(training_data_);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  double fitness = regressor.EvaluateIndividualFitness(sum_equation_);
  ASSERT_NEAR(fitness, 2.5, 1e-10);
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, EvaluateIndividualFitnessRelative) {
  ExplicitRegression regressor(training_data_, "mae", true);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  double fitness = regressor.EvaluateIndividualFitness(sum_equation_);
  ASSERT_NEAR(fitness, 1.0, 1e-10);
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, EvaluateIndividualFitnessWithNaN) {
  training_data_->x(0, 0) = std::numeric_limits<double>::quiet_NaN();
  ExplicitRegression regressor(training_data_);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  double fitness = regressor.EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(std::isnan(fitness));
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, GetIndividualFitnessAndGradient) {
  ExplicitRegression regressor(training_data_);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  Eigen::ArrayXd expected_gradient = Eigen::ArrayXd::Constant(5, 1, 1.0);

  double fitness;
  Eigen::ArrayXd gradient;
  std::tie(fitness, gradient) = regressor.GetIndividualFitnessAndGradient(sum_equation_);

  ASSERT_NEAR(fitness, 2.5, 1e-10);
  ASSERT_TRUE(expected_gradient.isApprox(gradient));
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, GetIndividualFitnessAndGradientRelative) {
  ExplicitRegression regressor(training_data_, "mae", true);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  Eigen::ArrayXd expected_gradient = Eigen::ArrayXd::Constant(5, 1, 1.0/2.5);

  double fitness;
  Eigen::ArrayXd gradient;
  std::tie(fitness, gradient) = regressor.GetIndividualFitnessAndGradient(sum_equation_);

  ASSERT_NEAR(fitness, 1.0, 1e-10);
  ASSERT_TRUE(expected_gradient.isApprox(gradient));
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, GetFitnessVectorAndJacobian) {
  ExplicitRegression regressor(training_data_);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  Eigen::ArrayXd expected_fitness_vector = Eigen::ArrayXd::Constant(10, 1, 2.5);
  Eigen::ArrayXXd expected_jacobian = training_data_->x;

  Eigen::ArrayXd fitness_vector;
  Eigen::ArrayXXd jacobian;
  std::tie(fitness_vector, jacobian) = regressor.GetFitnessVectorAndJacobian(sum_equation_);

  ASSERT_TRUE(expected_fitness_vector.isApprox(fitness_vector));
  ASSERT_TRUE(expected_jacobian.isApprox(jacobian));
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, GetFitnessVectorAndJacobianRelative) {
  ExplicitRegression regressor(training_data_, "mae", true);
  ASSERT_EQ(regressor.GetEvalCount(), 0);
  Eigen::ArrayXd expected_fitness_vector = Eigen::ArrayXd::Constant(10, 1, 1.0);
  Eigen::ArrayXXd expected_jacobian = Eigen::ArrayXXd::Constant(10, 5, 1.0/2.5);

  Eigen::ArrayXd fitness_vector;
  Eigen::ArrayXXd jacobian;
  std::tie(fitness_vector, jacobian) = regressor.GetFitnessVectorAndJacobian(sum_equation_);

  ASSERT_TRUE(expected_fitness_vector.isApprox(fitness_vector));
  ASSERT_TRUE(expected_jacobian.isApprox(jacobian));
  ASSERT_EQ(regressor.GetEvalCount(), 1);
}

TEST_F(TestExplicitRegression, GetSubsetOfTrainingData) {
  Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(5, 0, 4);
  ExplicitTrainingData* training_data = new ExplicitTrainingData(data_input, data_input);
  ExplicitTrainingData* subset_training_data = training_data->GetItem(std::vector<int>{0, 2, 3});

  Eigen::ArrayXXd expected_subset(3, 1);
  expected_subset << 0, 2, 3;
  ASSERT_TRUE(subset_training_data->x.isApprox(expected_subset));
  ASSERT_TRUE(subset_training_data->y.isApprox(expected_subset));
  delete training_data,
  delete subset_training_data;
}

TEST_F(TestExplicitRegression, CorrectTrainingDataSize) {
  for (int size : std::vector<int> {2, 5, 50}) {
    Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(size, 0, 10);
    ExplicitTrainingData* training_data = new ExplicitTrainingData(data_input, data_input);
    ASSERT_EQ(training_data->Size(), size);
    delete training_data;
  }
}

TEST_F(TestExplicitRegression, DumpLoadTrainingData) {
  ExplicitTrainingData* training_data_copy = new ExplicitTrainingData(training_data_->DumpState());

  ASSERT_TRUE(testutils::almost_equal(training_data_->x,
                                      training_data_copy->x));
  ASSERT_TRUE(testutils::almost_equal(training_data_->y,
                                      training_data_copy->y));

}

TEST_F(TestExplicitRegression, DumpLoadRegression) {
  ExplicitRegression regressor(training_data_);
  regressor.SetEvalCount(123);
  ExplicitRegression regressor_copy = ExplicitRegression(regressor.DumpState());

  ASSERT_EQ(regressor_copy.GetEvalCount(), 123);

}
} // namespace 