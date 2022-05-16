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
#ifndef INCLUDE_BINGOCPP_EVALUATION_BACKEND_H_
#define INCLUDE_BINGOCPP_EVALUATION_BACKEND_H_

#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <bingocpp/agraph/agraph.h>

namespace bingo {
/**
 * @brief This file contains the backend of the acyclic graph class.
 * 
 */
namespace evaluation_backend {
/**
 * @brief Evauluate the equation.
 * 
 * Evauluate the equation associated with an Agraph, at the values x.
 * 
 * @param stack Nx3 array. The command stack associated with an equation. 
 * N is the number of commands in the stack.
 * 
 * @param x MxD Array. Values at which to evaluate the equations. D is the
 * dimension in x and M is the number of data points in x.
 * 
 * @param constants Vector of doubles. Constants that are used in the equation.
 * 
 * @return Eigen::ArrayXXd The evaluation of the graph with x as the input data.
 */
Eigen::ArrayXXd Evaluate(const Eigen::ArrayX3i &stack,
                         const Eigen::ArrayXXd &x,
                         const Eigen::VectorXd &constants);

/**
 * @brief Evaluate equation and take derivative.
 * 
 * Evaluate the derivatives of the equation associated with an Agraph, 
 * at the values x.
 * 
 * @param stack Nx3 array. The command stack associated with an equation. 
 * N is the number of commands in the stack.
 * 
 * @param x MxD Array. Values at which to evaluate the equations. D is the
 * dimension in x and M is the number of data points in x.
 * 
 * @param constants Vector of doubles. Constants that are used in the equation.
 * 
 * @param param_x_or_c true: x derivative, false: c derivative
 * 
 * @return EvalAndDerivative Derivatives of all dimensions of x/constants at location x.
 */
EvalAndDerivative EvaluateWithDerivative(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const bool param_x_or_c = true);


} // namespace evaluation_backend
} // namespace bingo
#endif
