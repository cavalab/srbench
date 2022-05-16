/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/equation.h>
#include <python/py_equation.h>

namespace py = pybind11;
using namespace bingo;

void add_agraph_class(py::module &parent) {
  py::class_<Equation, bingo::PyEquation /* <---trampoline */>(parent, "Equation")
    .def(py::init<>())
    .def("evaluate_equation_at",
         &bingo::Equation::EvaluateEquationAt,
         py::arg("x"))
    .def("evaluate_equation_with_x_gradient_at",
         &bingo::Equation::EvaluateEquationWithXGradientAt,
         py::arg("x"))
    .def("evaluate_equation_with_local_opt_gradient_at",
         &bingo::Equation::EvaluateEquationWithLocalOptGradientAt,
         py::arg("x"))
    .def("get_complexity", &bingo::Equation::GetComplexity);

  py::class_<AGraph, bingo::Equation>(parent, "AGraph")
    .def(py::init<bool>(), py::arg("use_simplification")=false)
    .def_property_readonly_static("engine", [](py::object /* self */) { return "c++"; })
    .def_property("command_array",
                  &AGraph::GetCommandArray,
                  &AGraph::SetCommandArray)
    .def_property("mutable_command_array",
                  &AGraph::GetCommandArrayModifiable,
                  &AGraph::SetCommandArray)
    .def_property("fitness",
                  &AGraph::GetFitness,
                  &AGraph::SetFitness)
    .def_property("fit_set",
                  &AGraph::IsFitnessSet,
                  &AGraph::SetFitnessStatus)
    .def_property("genetic_age",
                  &AGraph::GetGeneticAge,
                  &AGraph::SetGeneticAge)
    .def_property("constants",
                  //&AGraph::GetLocalOptimizationParams,
                  [](AGraph& self) {
                    py::tuple tuple = py::cast(self.GetLocalOptimizationParams());
                    return tuple;
                  },
                  &AGraph::SetLocalOptimizationParams)
    .def("needs_local_optimization", &AGraph::NeedsLocalOptimization)
    .def("get_utilized_commands", &AGraph::GetUtilizedCommands)
    .def("get_number_local_optimization_params",
        &AGraph::GetNumberLocalOptimizationParams)
    .def("get_local_optimization_params",
        &AGraph::GetLocalOptimizationParams)
    .def("set_local_optimization_params", &AGraph::SetLocalOptimizationParams, py::arg("params"))
    .def("evaluate_equation_at", &AGraph::EvaluateEquationAt, py::arg("x"))
    .def("evaluate_equation_with_x_gradient_at",
        &AGraph::EvaluateEquationWithXGradientAt,
        py::arg("x"))
    .def("evaluate_equation_with_local_opt_gradient_at",
        &AGraph::EvaluateEquationWithLocalOptGradientAt,
        py::arg("x"))
    .def("__str__", &AGraph::GetConsoleString)
    .def("get_formatted_string", &AGraph::GetFormattedString,
         py::arg("format_"), py::arg("raw")=false)
    .def("get_complexity", &AGraph::GetComplexity)
    .def("distance", &AGraph::Distance, py::arg("chromosome"))
    .def("copy", &AGraph::Copy)
    .def("__getstate__", &AGraph::DumpState)
    .def("__setstate__", [](AGraph &ag, const AGraphState &state) {
            new (&ag) AGraph(state); });
}