#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <bingocpp/equation.h>
#include <bingocpp/fitness_function.h>
#include <bingocpp/training_data.h>

#include "test_fixtures.h"
#include "testing_utils.h"

namespace {

const double pi = std::acos(-1);

struct SampleTrainingData : public bingo::TrainingData {
  Eigen::ArrayXXd x;
  Eigen::ArrayXXd y;
  SampleTrainingData() : TrainingData() {}
  ~SampleTrainingData() {}
  SampleTrainingData(Eigen::ArrayXXd &x, Eigen::ArrayXXd &y) : 
      x(x), y(y) { }
  SampleTrainingData* GetItem(int) {
    throw new std::logic_error("Not implemented Exception");
  }
  SampleTrainingData* GetItem(const std::vector<int> &) {
    throw new std::logic_error("Not implemented Exception");
  }
  int Size() { return x.rows(); }
};

class SampleFitnessFunction : public bingo::VectorBasedFunction {
 public:
  SampleFitnessFunction(SampleTrainingData* training_data,
      std::string metric = "mae") :
      bingo::VectorBasedFunction(training_data, metric) {}
  ~SampleFitnessFunction() {} 
  Eigen::ArrayXd EvaluateFitnessVector(bingo::Equation &individual) const {
    Eigen::ArrayXXd f_of_x =
        individual.EvaluateEquationAt(((SampleTrainingData*)training_data_)->x);
    return f_of_x - ((SampleTrainingData*)training_data_)->y;
  }
};

class TestFitnessFunction : public testing::Test {
 public:
  SampleTrainingData training_data_;
  SampleFitnessFunction* sample_fitness_function_;
  void SetUp() {
    training_data_ = init_sample_training_data();
    sample_fitness_function_ = new SampleFitnessFunction(&training_data_);
  }

  void TearDown() {
    delete sample_fitness_function_;
  }

 private:
  SampleTrainingData init_sample_training_data() {
    Eigen::ArrayXXd x(3,3);
    x << ((1./6.) * pi), 1., 1.,
         ((1./2.) * pi), 3., 4.,
         pi            , 9., 16.;
    Eigen::ArrayXXd y(3, 1);
    y << 1.5, 2, 1;
    SampleTrainingData training_data(x, y);
    return training_data;
  }
};

TEST_F(TestFitnessFunction, InvalidTrainingMetric) {
  try {
    SampleFitnessFunction test_function(&training_data_, "invalid_metric");
    FAIL() << "Expecting std::invalid_argument exception\n";
  } catch (std::invalid_argument &exception) {
    SUCCEED();
  }
}

TEST_F(TestFitnessFunction, CorrectMetricSet) {
  SampleFitnessFunction mse(&training_data_, "mse");
  SampleFitnessFunction rmse(&training_data_, "rmse");
  bingo::AGraph agraph = testutils::init_sample_agraph_1();
  double mae_expected = 0.600018;
  double mse_expected = 0.389427;
  double rmse_expected = 0.624041;
  double error_tol = 10e-6;
  ASSERT_NEAR(mae_expected,
              sample_fitness_function_->EvaluateIndividualFitness(agraph),
              error_tol);
  ASSERT_NEAR(mse_expected,
              mse.EvaluateIndividualFitness(agraph),
              error_tol);
  ASSERT_NEAR(rmse_expected,
              rmse.EvaluateIndividualFitness(agraph),
              error_tol);
}
} // namespace (anonymous)