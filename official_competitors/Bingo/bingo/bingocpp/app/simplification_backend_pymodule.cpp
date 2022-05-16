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

#include <bingocpp/agraph/simplification_backend/simplification_backend.h>

namespace py = pybind11;
using namespace bingo;

void add_simplification_backend_submodule(py::module &parent) {
  py::module m = parent.def_submodule("simplification_backend",
                                      "The simplification backend for Agraphs");
  m.attr("ENGINE") = "c++";
  m.def("get_utilized_commands", &simplification_backend::GetUtilizedCommands,
        "Find which commands are utilized",
        py::arg("stack"));
  m.def("simplify_stack", &simplification_backend::PythonSimplifyStack,
        "Simplifies a stack based on computational algebra",
        py::arg("stack"));
  m.def("reduce_stack", &simplification_backend::SimplifyStack, "Reduces a stack",
        py::arg("stack"));

}