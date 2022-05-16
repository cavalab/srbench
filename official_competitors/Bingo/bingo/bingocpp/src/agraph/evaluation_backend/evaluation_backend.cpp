#include <map>
#include <numeric>

#include <Eigen/Dense>

#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/evaluation_backend/operator_eval.h>
#include <bingocpp/agraph/constants.h>
#include <bingocpp/agraph/operator_definitions.h>

namespace bingo {
namespace evaluation_backend {
namespace {

Eigen::ArrayXXd reverse_eval(const std::pair<int, int> &deriv_shape,
                             const int deriv_wrt_node,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             const Eigen::ArrayX3i &stack);

std::vector<Eigen::ArrayXXd> forward_eval(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants);

EvalAndDerivative evaluate_with_derivative(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const bool param_x_or_c);
} // namespace

Eigen::ArrayXXd Evaluate(const Eigen::ArrayX3i &stack,
                         const Eigen::ArrayXXd &x,
                         const Eigen::VectorXd &constants) {
  std::vector<Eigen::ArrayXXd> _forward_eval = forward_eval(
      stack, x, constants);
  return _forward_eval.back();  
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const bool param_x_or_c) {
  return evaluate_with_derivative(
    stack, x, constants, param_x_or_c);
}


namespace {

Eigen::ArrayXXd reverse_eval(const std::pair<int, int> &deriv_shape,
                             const int deriv_wrt_node,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             const Eigen::ArrayX3i &stack) {
  int num_samples = deriv_shape.first;
  int num_features = deriv_shape.second;
  int stack_depth = stack.rows();

  Eigen::ArrayXXd derivative = Eigen::ArrayXXd::Zero(num_samples, num_features);
  std::vector<Eigen::ArrayXXd> reverse_eval(stack_depth); 
  for (int row = 0; row < stack_depth; row++) {
      reverse_eval[row] = Eigen::ArrayXd::Zero(num_samples);
  }

  reverse_eval[stack_depth-1] = Eigen::ArrayXd::Ones(num_samples);
  for (int i = stack_depth - 1; i >= 0; i--) {
    int node = stack(i, kOpIdx);
    int param1 = stack(i, kParam1Idx);
    int param2 = stack(i, kParam2Idx);
    if (node == deriv_wrt_node) {
      derivative.col(param1) += reverse_eval[i];
    } else {
      ReverseEvalFunction(node, i, param1, param2, forward_eval, reverse_eval);
    }
  }
  return derivative;
}

std::vector<Eigen::ArrayXXd> forward_eval(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants) {
  std::vector<Eigen::ArrayXXd> _forward_eval(stack.rows());

  for (int i = 0; i < stack.rows(); ++i) {
    int node = stack(i, kOpIdx);
    int op1 = stack(i, kParam1Idx);
    int op2 = stack(i, kParam2Idx);
    _forward_eval[i] = ForwardEvalFunction(
      node, op1, op2, x, constants, _forward_eval);
  }
  return _forward_eval;
}

EvalAndDerivative evaluate_with_derivative(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const bool param_x_or_c) {
  std::vector<Eigen::ArrayXXd> _forward_eval = forward_eval(
      stack, x, constants);

  std::pair<int, int> deriv_shape;
  int deriv_wrt_node;
  if (param_x_or_c) {  // true = x
    deriv_shape = std::make_pair(x.rows(), x.cols());
    deriv_wrt_node = Op::kVariable;
  } else {  // false = c
    deriv_shape = std::make_pair(x.rows(), constants.size());
    deriv_wrt_node = Op::kConstant;
  }

  Eigen::ArrayXXd derivative = reverse_eval(
      deriv_shape, deriv_wrt_node, _forward_eval, stack);
  return std::make_pair(_forward_eval.back(), derivative);
}

} // namespace (anonymous)
} // namespace backend
} // namespace bingo 
