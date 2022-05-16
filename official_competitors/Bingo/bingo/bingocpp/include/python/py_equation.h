#ifndef BINGOCPP_INCLUDE_BINGOCPP_PY_EQUATION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_PY_EQUATION_H_

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "bingocpp/equation.h"

namespace bingo {

class PyEquation : public Equation {
 public:
  Eigen::ArrayXXd 
  EvaluateEquationAt(const Eigen::ArrayXXd &x) {
    PYBIND11_OVERLOAD_PURE_NAME(
      Eigen::ArrayXXd,
      Equation,
      "evaluate_equation_at",
      EvaluateEquationAt,
      x
    );
  }

  EvalAndDerivative
  EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x) {
    PYBIND11_OVERLOAD_PURE_NAME(
      EvalAndDerivative,
      Equation,
      "evaluate_equation_with_x_gradient_at",
      EvaluateEquationWithXGradientAt,
      x
    );
  }

  EvalAndDerivative
  EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x) {
    PYBIND11_OVERLOAD_PURE_NAME(
      EvalAndDerivative,
      Equation,
      "evaluate_equation_with_local_opt_gradient_at",
      EvaluateEquationWithLocalOptGradientAt,
      x
    );
  }

  int GetComplexity() override {
    PYBIND11_OVERLOAD_PURE_NAME(
      int,
      Equation,
      "get_complexity",
      GetComplexity,
    );
  }
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_PY_EQUATION_H_