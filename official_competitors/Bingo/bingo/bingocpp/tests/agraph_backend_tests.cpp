#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/simplification_backend/simplification_backend.h>

#include "testing_utils.h"
#include "test_fixtures.h"

using namespace bingo;
using namespace evaluation_backend;
using namespace simplification_backend;

namespace {

const int N_OPS = 13;

struct AGraphValues {
Eigen::ArrayXXd x_vals;
Eigen::VectorXd constants;
AGraphValues() {}
AGraphValues(Eigen::ArrayXXd &x, Eigen::VectorXd &c) : 
  x_vals(x), constants(c) {}
};

class AGraphBackend : public ::testing::TestWithParam<int> {
 public:
  
  const double AGRAPH_VAL_START =-1;
  const double AGRAPH_VAL_END = 0;
  const int N_AGRAPH_VAL = 11;

  AGraphValues sample_agraph_1_values;
  std::vector<Eigen::ArrayXXd> operator_evals_x0;
  std::vector<Eigen::ArrayXXd> operator_x_derivs;
  std::vector<Eigen::ArrayXXd> operator_c_derivs;

  Eigen::ArrayX3i simple_stack;
  Eigen::ArrayX3i simple_stack2;
  Eigen::ArrayXXd x;
  Eigen::ArrayXd constants;

  virtual void SetUp() {
    sample_agraph_1_values = init_agraph_vals(AGRAPH_VAL_START,
                                              AGRAPH_VAL_END,
                                              N_AGRAPH_VAL);           
    operator_evals_x0 = init_op_evals_x0(sample_agraph_1_values);
    operator_x_derivs = init_op_x_derivs(sample_agraph_1_values);
    operator_c_derivs = init_op_c_derivs(sample_agraph_1_values);

    simple_stack = testutils::stack_operators_0_to_5(); 
    simple_stack2 = testutils::stack_unary_operator(4);
    x = testutils::one_to_nine_3_by_3();
    constants = testutils::pi_ten_constants();
  }
  virtual void TearDown() {}

 AGraphValues init_agraph_vals(double begin, double end, int num_points) {
  Eigen::VectorXd constants = Eigen::VectorXd(2);
  constants << 10, 3.14;

  Eigen::ArrayXXd x_vals(num_points, 2);
  x_vals.col(0) = Eigen::ArrayXd::LinSpaced(num_points, begin, 0);
  x_vals.col(1) = Eigen::ArrayXd::LinSpaced(num_points, 0, end);

  return AGraphValues(x_vals, constants);
}

std::vector<Eigen::ArrayXXd> init_op_evals_x0(
    const AGraphValues &sample_agraph_1_values) {
  Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
  double constant = sample_agraph_1_values.constants[0];
  Eigen::ArrayXXd c_0 = constant * Eigen::ArrayXd::Ones(x_0.rows());

  const std::vector<Eigen::ArrayXXd> op_evals_x0 = {
    x_0,
    c_0,
    x_0+x_0,
    x_0-x_0,
    x_0*x_0,
    x_0/x_0,
    x_0.sin(),
    x_0.cos(),
    x_0.exp(),
    x_0.abs().log(),
    x_0.abs().pow(x_0),
    x_0.abs(),
    x_0.abs().sqrt()
  };

  return op_evals_x0;
}

std::vector<Eigen::ArrayXXd> init_op_x_derivs(
    const AGraphValues &sample_agraph_1_values) {
  Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
  int size = x_0.rows();

  auto last_nan = [](Eigen::ArrayXXd array) {
    array(array.rows() - 1, array.cols() -1) = std::nan("1");
    Eigen::ArrayXXd modified_array = array;
    return modified_array;
  };
  
  const std::vector<Eigen::ArrayXXd> op_x_derivs = {
    Eigen::ArrayXd::Ones(size),
    Eigen::ArrayXd::Zero(size),
    2.0  * Eigen::ArrayXd::Ones(size),
    Eigen::ArrayXd::Zero(size),
    2.0 * x_0,
    last_nan(Eigen::ArrayXd::Zero(size)),
    x_0.cos(),
    -x_0.sin(),
    x_0.exp(),
    1.0 / x_0,
    last_nan(x_0.abs().pow(x_0)*(x_0.abs().log() + Eigen::ArrayXd::Ones(size))),
    x_0.sign(),
    0.5 * x_0.sign() / x_0.abs().sqrt()
  };
  
  return op_x_derivs;
}

std::vector<Eigen::ArrayXXd> init_op_c_derivs(
    const AGraphValues &sample_agraph_1_values) {
  int size = sample_agraph_1_values.x_vals.rows();
  Eigen::ArrayXXd c_1 = sample_agraph_1_values.constants[1] * Eigen::ArrayXd::Ones(size);

  std::vector<Eigen::ArrayXXd> op_c_derivs =  {
    Eigen::ArrayXd::Zero(size),
    Eigen::ArrayXd::Ones(size),
    2.0  * Eigen::ArrayXd::Ones(size),
    Eigen::ArrayXd::Zero(size),
    2.0 * c_1,
    (Eigen::ArrayXd::Zero(size)),
    c_1.cos(),
    -c_1.sin(),
    c_1.exp(),
    1.0 / c_1,
    c_1.abs().pow(c_1)*(c_1.abs().log() + Eigen::ArrayXd::Ones(size)),
    c_1.sign(),
    0.5 * c_1.sign() / c_1.abs().sqrt()
  };
  
  return op_c_derivs;
}
};

INSTANTIATE_TEST_CASE_P(,AGraphBackend, ::testing::Range(0, N_OPS, 1));

TEST_F(AGraphBackend, evaluate) {
  Eigen::ArrayXXd y = Evaluate(simple_stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] 
                          / x.col(1)) - x.col(0);
  ASSERT_TRUE(testutils::almost_equal(y, y_true));
}

TEST_F(AGraphBackend, evaluate_and_derivative) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    EvaluateWithDerivative(simple_stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] 
                          / x.col(1)) - x.col(0);
  Eigen::ArrayXXd dy_true = Eigen::ArrayXXd::Zero(3, 3);
  dy_true.col(0) = constants[0] + constants[1] / x.col(1) - 1.;
  dy_true.col(1) = - x.col(0) * constants[1] / x.col(1) / x.col(1);

  ASSERT_TRUE(testutils::almost_equal(y_and_dy.first, y_true));
  ASSERT_TRUE(testutils::almost_equal(y_and_dy.second, dy_true));
}

TEST_F(AGraphBackend, get_utilized_commands) {
  std::vector<bool> used_commands = GetUtilizedCommands(simple_stack);
  int num_used_commands = 0;
  for (auto const &command_is_used : used_commands) {
    if (command_is_used) {
      ++num_used_commands;
    }
  }
  ASSERT_EQ(num_used_commands, 8);
}
} // namespace
