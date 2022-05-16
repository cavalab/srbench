#include <cmath>

#include "gtest/gtest.h"
#include <Eigen/Dense>

#include <bingocpp/implicit_regression.h>

#include "test_fixtures.h"

using namespace bingo;

namespace {

class ImplicitRegressionFixture {
 public:
  ImplicitTrainingData *training_data_;
  testutils::SumEquation sum_equation_;

 protected:
  ImplicitTrainingData* init_sample_training_data() {
    const int num_points = 50;
    const int num_data_per_feature = 10;
    const int num_feature = 50 / num_data_per_feature;
    const int index_block_mod = 3;

    Eigen::ArrayXXd x = Eigen::ArrayXd::LinSpaced(num_points, 0, 0.98);
    x = x.reshaped(num_feature, num_data_per_feature);
    x.transposeInPlace();

    Eigen::ArrayXXd dx_dt = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), 1);
    dx_dt.block(
        0, index_block_mod, dx_dt.rows(), dx_dt.cols() - index_block_mod)
        = Eigen::ArrayXXd::Constant(x.rows(), 2, -1);
    dx_dt.col(dx_dt.cols()/2) = Eigen::ArrayXd::Constant(dx_dt.rows(), 0);
    return new ImplicitTrainingData(x, dx_dt);
  }
};

class ImplicitRegressionTestNormalize : public ImplicitRegressionFixture,
                                        public testing::TestWithParam<bool> {
 public:
  virtual void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  virtual void TearDown() {
    delete training_data_;
  }
};

TEST_P(ImplicitRegressionTestNormalize, EvaluateIndividualFitness) {
  auto regressor = new ImplicitRegression(training_data_, -1);
  double fitness = regressor->EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(0.14563031020 - fitness < 1e-10);
  delete regressor;
}
INSTANTIATE_TEST_CASE_P(,ImplicitRegressionTestNormalize, testing::Bool());

class ImplicitRegressionTestNonNormalized : 
    public ImplicitRegressionFixture,
    public testing::TestWithParam<std::tuple<int, bool>> {
 public:
  virtual void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  virtual void TearDown() {
    delete training_data_;
  }
};

TEST_P(ImplicitRegressionTestNonNormalized, EvaluateIndividualFitness) {
  auto const &param = GetParam();
  auto required_params = std::get<0>(param);
  auto infinite_fitness = std::get<1>(param);
  auto regressor = new ImplicitRegression(training_data_, required_params);
  double fitness = regressor->EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(!std::isfinite(fitness) == infinite_fitness);
  delete regressor;
}
INSTANTIATE_TEST_CASE_P(instance_one, ImplicitRegressionTestNonNormalized,
  ::testing::Values(std::make_tuple(4, false), std::make_tuple(5, true))
);

class ImplicitRegressionTest : public ImplicitRegressionFixture,
                               public testing::Test {
 public:
  virtual void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  virtual void TearDown() {
    delete training_data_;
  }
};

TEST_F(ImplicitRegressionTest, GetSubsetOfData) {
  auto data_input = Eigen::ArrayXd::LinSpaced(5, 0, 4);
  auto training_data = new ImplicitTrainingData(data_input, data_input);
  auto subset_training_data = training_data->GetItem(std::vector<int>{0, 2, 3});
  Eigen::ArrayXXd expected_subset(3, 1);
  expected_subset << 0, 2, 3;
  ASSERT_TRUE(subset_training_data->x.isApprox(expected_subset));
  ASSERT_TRUE(subset_training_data->dx_dt.isApprox(expected_subset));
  delete training_data;
  delete subset_training_data;
}

TEST_F(ImplicitRegressionTest, CorrectTrainingDataSize) {
  for (int size : std::vector<int> {2, 5, 50}) {
    Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(size, 0, 10);
    auto training_data = new ImplicitTrainingData(data_input, data_input);
    ASSERT_EQ(training_data->Size(), size);
    delete training_data;
  }
}

TEST_F(ImplicitRegressionTest, DumpLoadTrainingData) {
  ImplicitTrainingData* training_data_copy = new ImplicitTrainingData(training_data_->DumpState());

  ASSERT_TRUE(testutils::almost_equal(training_data_->x,
                                      training_data_copy->x));
  ASSERT_TRUE(testutils::almost_equal(training_data_->dx_dt,
                                      training_data_copy->dx_dt));

}

TEST_F(ImplicitRegressionTest, DumpLoadRegression) {
  ImplicitRegression regressor(training_data_);
  regressor.SetEvalCount(123);
  ImplicitRegression regressor_copy = ImplicitRegression(regressor.DumpState());

  ASSERT_EQ(regressor_copy.GetEvalCount(), 123);

}

TEST(ImplicitRegressionPartials, PartialCalculationInTrainingData) {
  auto data_input = Eigen::ArrayXd::LinSpaced(20, 0., 19.);
  Eigen::ArrayXXd data_array(data_input.rows(), 3);
  data_array << data_input * 0, data_input * 1, data_input * 2;
  auto training_data = new ImplicitTrainingData(data_array);
  Eigen::ArrayXXd expected_derivatives(13, 3);
  expected_derivatives << Eigen::ArrayXd::Ones(13) * 0,
                          Eigen::ArrayXd::Ones(13) * 1,
                          Eigen::ArrayXd::Ones(13) * 2;
  ASSERT_TRUE(training_data->dx_dt.isApprox(expected_derivatives));
  delete training_data;
}

TEST(ImplicitRegressionPartials, PartialCalculationInTrainingDataNaN) {
  auto data_input = Eigen::ArrayXd::LinSpaced(20, 0., 19.) * 2;
  Eigen::ArrayXXd data_array(data_input.rows() * 2 + 1, 1);
  data_array << data_input,
                std::numeric_limits<double>::quiet_NaN(),
                data_input;
  auto training_data = new ImplicitTrainingData(data_array);
  Eigen::ArrayXd expected_derivative = Eigen::ArrayXd::Constant(26, 2.0);
  ASSERT_TRUE(training_data->dx_dt.isApprox(expected_derivative));
  delete training_data;
}
} // namespace (anonymous)