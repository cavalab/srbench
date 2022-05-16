#include <stdexcept>

#include <bingocpp/agraph/evaluation_backend/operator_eval.h>
#include <bingocpp/agraph/operator_definitions.h>

namespace bingo {
namespace evaluation_backend {
namespace { 

// Integer
Eigen::ArrayXXd integer_forward_eval(int param1, int,
                                     const Eigen::ArrayXXd &x,
                                     const Eigen::VectorXd &,
                                     std::vector<Eigen::ArrayXXd> &) {
  return Eigen::ArrayXd::Constant(x.rows(), param1);
}

void integer_reverse_eval(int, int, int,
                          const std::vector<Eigen::ArrayXXd> &,
                          std::vector<Eigen::ArrayXXd> &) {
  return;
}

// Load x
Eigen::ArrayXXd loadx_forward_eval(int param1, int,
                                   const Eigen::ArrayXXd &x,
                                   const Eigen::VectorXd &,
                                   std::vector<Eigen::ArrayXXd> &) {
  return x.col(param1);
}

void loadx_reverse_eval(int, int, int,
                        const std::vector<Eigen::ArrayXXd> &,
                        std::vector<Eigen::ArrayXXd> &) {
  return;
}

// Load c
Eigen::ArrayXXd loadc_forward_eval(int param1, int, 
                                   const Eigen::ArrayXXd &x, 
                                   const Eigen::VectorXd &constants, 
                                   std::vector<Eigen::ArrayXXd> &) {
  return Eigen::ArrayXd::Constant(x.rows(), constants[param1]);
}

void loadc_reverse_eval(int, int, int,
                        const std::vector<Eigen::ArrayXXd> &,
                        std::vector<Eigen::ArrayXXd> &) {
  return;
}

// Addition
Eigen::ArrayXXd add_forward_eval(int param1, int param2, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] + forward_eval[param2]; 
} 

void add_reverse_eval(int reverse_index, int param1, int param2, 
                      const std::vector<Eigen::ArrayXXd> &, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index];
  reverse_eval[param2] += reverse_eval[reverse_index];
} 

// Subtraction
Eigen::ArrayXXd subtract_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &,
                                      const Eigen::VectorXd &, 
                                      std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] - forward_eval[param2]; 
} 

void subtract_reverse_eval(int reverse_index, int param1, int param2, 
                           const std::vector<Eigen::ArrayXXd> &, 
                           std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index];
  reverse_eval[param2] -= reverse_eval[reverse_index];
}

// Multiplication
Eigen::ArrayXXd multiply_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &, 
                                      const Eigen::VectorXd &, 
                                      std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] * forward_eval[param2]; 
} 

void multiply_reverse_eval(int reverse_index, int param1, int param2, 
                           const std::vector<Eigen::ArrayXXd> &forward_eval, 
                           std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                              *forward_eval[param2];
  reverse_eval[param2] += reverse_eval[reverse_index]
                              *forward_eval[param1];
} 

// Division
Eigen::ArrayXXd divide_forward_eval(int param1, int param2, 
                                    const Eigen::ArrayXXd &, 
                                    const Eigen::VectorXd &, 
                                    std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] / forward_eval[param2]; 
} 

void divide_reverse_eval(int reverse_index, int param1, int param2, 
                         const std::vector<Eigen::ArrayXXd> &forward_eval, 
                         std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                              /forward_eval[param2];
  reverse_eval[param2] -= reverse_eval[reverse_index]
                              *forward_eval[reverse_index]
                              /forward_eval[param2];
}

// Sine
Eigen::ArrayXXd sin_forward_eval(int param1, int, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval.at(param1).sin(); 
}

void sin_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].cos();
}

// Cosine
Eigen::ArrayXXd cos_forward_eval(int param1, int, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].cos(); 
}

void cos_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] -= reverse_eval[reverse_index]
                         *forward_eval[param1].sin();
}

// Exponential 
Eigen::ArrayXXd exp_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].exp();
}

void exp_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index];
}

// Logarithm
Eigen::ArrayXXd log_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs().log();
}

void log_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         /forward_eval[param1];
}

// Power
Eigen::ArrayXXd pow_forward_eval(int param1, int param2,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].pow(forward_eval[param2]);
}

void pow_reverse_eval(int reverse_index, int param1, int param2, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *forward_eval[param2]
                         /forward_eval[param1];
  reverse_eval[param2] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *(forward_eval[param1].log());
}

// Safe Power
Eigen::ArrayXXd safepow_forward_eval(int param1, int param2,
                                     const Eigen::ArrayXXd &,
                                     const Eigen::VectorXd &,
                                     std::vector<Eigen::ArrayXXd> &forward_eval
                                     ) {
  return forward_eval[param1].abs().pow(forward_eval[param2]);
}

void safepow_reverse_eval(int reverse_index, int param1, int param2,
                          const std::vector<Eigen::ArrayXXd> &forward_eval,
                          std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *forward_eval[param2]
                         /forward_eval[param1];
  reverse_eval[param2] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *(forward_eval[param1].abs().log());
}

// Absolute Value
Eigen::ArrayXXd abs_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs();
}

void abs_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].sign();
}

// Sqruare root
Eigen::ArrayXXd sqrt_forward_eval(int param1, int,
                                  const Eigen::ArrayXXd &,
                                  const Eigen::VectorXd &,
                                  std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs().sqrt();
}

void sqrt_reverse_eval(int reverse_index, int param1, int, 
                       const std::vector<Eigen::ArrayXXd> &forward_eval, 
                       std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += 0.5*reverse_eval[reverse_index]
                              /forward_eval[reverse_index]
                              *forward_eval[param1].sign();
}

// Sinh
Eigen::ArrayXXd sinh_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval.at(param1).sinh();
}

void sinh_reverse_eval(int reverse_index, int param1, int,
                      const std::vector<Eigen::ArrayXXd> &forward_eval,
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].cosh();
}

// Cosh
Eigen::ArrayXXd cosh_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].cosh();
}

void cosh_reverse_eval(int reverse_index, int param1, int,
                      const std::vector<Eigen::ArrayXXd> &forward_eval,
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].sinh();
}

} // namespace

Eigen::ArrayXXd ForwardEvalFunction(int node, int param1, int param2,
                                    const Eigen::ArrayXXd &x, 
                                    const Eigen::VectorXd &constants,
                                    std::vector<Eigen::ArrayXXd> &forward_eval
                                    ) {
  switch (node) {
    case Op::kInteger :
      return integer_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kVariable :
      return loadx_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kConstant :
      return loadc_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kAddition :
      return add_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kSubtraction :
      return subtract_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kMultiplication :
      return multiply_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kDivision :
      return divide_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kSin :
      return sin_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kCos :
      return cos_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kExponential :
      return exp_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kLogarithm :
      return log_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kPower :
      return pow_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kAbs :
      return abs_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kSqrt :
      return sqrt_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kSafePower :
      return safepow_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kSinh :
      return sinh_forward_eval(param1, param2, x, constants, forward_eval);
    case Op::kCosh :
      return cosh_forward_eval(param1, param2, x, constants, forward_eval);
  }
  throw std::runtime_error("Unknown Operator In Forward Evaluation");
}

void ReverseEvalFunction(int node, int reverse_index, int param1, int param2,
                         const std::vector<Eigen::ArrayXXd> &forward_eval,
                         std::vector<Eigen::ArrayXXd> &reverse_eval) {
  switch (node) {
    case Op::kInteger :
      return integer_reverse_eval(reverse_index, param1, param2, forward_eval,
                                  reverse_eval);
    case Op::kVariable :
      return loadx_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kConstant :
      return loadc_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kAddition :
      return add_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kSubtraction :
      return subtract_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kMultiplication :
      return multiply_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kDivision :
      return divide_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kSin :
      return sin_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kCos :
      return cos_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kExponential :
      return exp_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kLogarithm :
      return log_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kPower :
      return pow_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kAbs :
      return abs_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kSqrt :
      return sqrt_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kSafePower :
      return safepow_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kSinh :
      return sinh_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
    case Op::kCosh :
      return cosh_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
  }
  throw std::runtime_error("Unknown Operator In Reverse Evaluation");
}
} // namespace backend
} // namespace bingo