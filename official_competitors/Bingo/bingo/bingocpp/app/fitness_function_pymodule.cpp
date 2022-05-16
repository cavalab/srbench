#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <bingocpp/fitness_function.h>
#include <bingocpp/training_data.h>
#include <python/py_fitness_function.h>

namespace py = pybind11;
using namespace bingo;

void add_fitness_classes(py::module &parent) {
  py::class_<FitnessFunction, PyFitnessFunction /* trampoline */>(parent, "FitnessFunction")
    .def(py::init<TrainingData *>(),
         py::arg("training_data") = py::none())
    .def("__call__", &FitnessFunction::EvaluateIndividualFitness)
    .def_property("eval_count", &FitnessFunction::GetEvalCount, &FitnessFunction::SetEvalCount)
    .def_property("training_data", &FitnessFunction::GetTrainingData, &FitnessFunction::SetTrainingData);

  py::class_<TrainingData, PyTrainingData /* trampoline */>(parent, "TrainingData")
    .def(py::init<>())
    .def("__getitem__",
         static_cast<TrainingData* (TrainingData::*)(const std::vector<int> &)>(&TrainingData::GetItem),
         py::arg("items"))
    .def("__len__", &TrainingData::Size);

  py::class_<VectorBasedFunction, FitnessFunction, PyVectorBasedFunction /* trampoline */>(parent, "VectorBasedFunction")
    .def(py::init<TrainingData *, std::string>(),
         py::arg("training_data") = py::none(),
         py::arg("metric") = "mae")
    .def("__call__", &VectorBasedFunction::EvaluateIndividualFitness)
    .def("evaluate_fitness_vector", &VectorBasedFunction::EvaluateFitnessVector);
}