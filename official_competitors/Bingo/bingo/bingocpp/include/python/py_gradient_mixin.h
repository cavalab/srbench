#ifndef BINGOCPP_INCLUDE_BINGOCPP_PY_GRADIENT_MIXIN_H
#define BINGOCPP_INCLUDE_BINGOCPP_PY_GRADIENT_MIXIN_H

#include <bingocpp/gradient_mixin.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <tuple>

namespace py = pybind11;

namespace bingo {

class PyGradientMixin : public GradientMixin {
 public:
  FitnessAndGradient GetIndividualFitnessAndGradient(Equation &individual) {
    PYBIND11_OVERLOAD_PURE_NAME(
      FitnessAndGradient,
      GradientMixin,
      "get_fitness_and_gradient",
      GetIndividualFitnessAndGradient,
      individual
    );
  }
};

class PyVectorGradientMixin : public VectorGradientMixin {
 public:
  using VectorGradientMixin::VectorGradientMixin;

  FitnessAndGradient GetIndividualFitnessAndGradient(Equation &individual) const {
    PYBIND11_OVERLOAD_NAME(
      FitnessAndGradient,
      VectorGradientMixin,
      "get_fitness_and_gradient",
      GetIndividualFitnessAndGradient,
      individual
    );
  }

  virtual FitnessVectorAndJacobian GetFitnessVectorAndJacobian(Equation &individual) const {
    PYBIND11_OVERLOAD_PURE_NAME(
      FitnessVectorAndJacobian,
      VectorGradientMixin,
      "get_fitness_vector_and_jacobian",
      GetFitnessVectorAndJacobian,
      individual
    );
  }
};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_PY_GRADIENT_MIXIN_H
