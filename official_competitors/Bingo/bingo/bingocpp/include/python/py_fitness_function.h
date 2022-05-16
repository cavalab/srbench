#ifndef BINGOCPP_INCLUDE_BINGOCPP_PY_FITNESS_FUNCTION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_PY_FITNESS_FUNCTION_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <bingocpp/fitness_function.h>
#include <bingocpp/training_data.h>

namespace bingo {
  class PyTrainingData : public TrainingData {
    using TrainingData::TrainingData;
    TrainingData* GetItem(int item) {
      PYBIND11_OVERLOAD_PURE_NAME(
        TrainingData*,
        TrainingData,
        "__getitem__",
        GetItem,
        item
      );
    }

    TrainingData* GetItem(const std::vector<int> &items) {
      PYBIND11_OVERLOAD_PURE_NAME(
        TrainingData*,
        TrainingData,
        "__getitem__",
        GetItem,
        items
      );
    }

    int Size() {
      PYBIND11_OVERLOAD_PURE_NAME(
        int,
        TrainingData,
        "__len__",
        Size,
        // no arguments
      );
    }
  };

  class PyFitnessFunction : public FitnessFunction {
    using FitnessFunction::FitnessFunction;
    double EvaluateIndividualFitness(Equation &individual) const {
      PYBIND11_OVERLOAD_PURE_NAME(
        double,
        FitnessFunction,
        "__call__",
        EvaluateIndividualFitness,
        individual
      );
    }
  };

  class PyVectorBasedFunction : public VectorBasedFunction {
    using VectorBasedFunction::VectorBasedFunction;
    double EvaluateIndividualFitness(Equation &individual) const {
      PYBIND11_OVERLOAD_NAME(
        double,
        VectorBasedFunction,
        "__call__",
        EvaluateIndividualFitness,
        individual
      );
    }

    Eigen::ArrayXd EvaluateFitnessVector(Equation &individual) const {
      PYBIND11_OVERLOAD_PURE_NAME(
          Eigen::ArrayXd,
          VectorBasedFunction,
          "evaluate_fitness_vector",
          EvaluateFitnessVector,
          individual
      );
    }
  };

} // namespace bingo

#endif // BINGOCPP_INCLUDE_BINGOCPP_PY_FITNESS_FUNCTION_H_
