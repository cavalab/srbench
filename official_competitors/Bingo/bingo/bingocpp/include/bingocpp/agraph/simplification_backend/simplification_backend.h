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
#ifndef INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_H
#define INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_H

#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <bingocpp/agraph/agraph.h>

namespace bingo {
/**
 * @brief This file contains the simplification backend of the agraph class.
 * 
 */
namespace simplification_backend {
/**
 * @brief Simplifies a stack.
 *
 * An acyclic graph is given in stack form.  The stack is first simplified to
 * consist only of the commands used by the last command.
 *
 * @param stack Description of an acyclic graph in stack format.
 *
 * @return Simplified stack.
 */
Eigen::ArrayX3i SimplifyStack(const Eigen::ArrayX3i &stack);

// TODO documentation and change simplify_stack to reduce stack
Eigen::ArrayX3i PythonSimplifyStack(const Eigen::ArrayX3i &stack);

/**
 * @brief Finds which commands are utilized in a stack.
 *
 * An acyclic graph is given in stack form.  The stack is processed in reverse
 * to find which commands the last command depends.
 *
 * @param stack Description of an acyclic graph in stack format.
 *
 * @return vector describing which commands in the stack are used.
 */
std::vector<bool> GetUtilizedCommands(const Eigen::ArrayX3i &stack);
} // namespace simplification_backend
} // namespace bingo
#endif
