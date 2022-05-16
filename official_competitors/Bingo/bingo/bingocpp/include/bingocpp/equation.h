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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_

#include <string>

#include <Eigen/Dense>

namespace bingo {
 
typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvalAndDerivative;

class Equation {
 public:
   virtual ~Equation() = default;
   
   /**
   * @brief Evaluate the Equation
   * 
   * Evaluation of the Equation at points x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return Eigen::ArrayXXd The evaluation of function at points x.
   */
  virtual Eigen::ArrayXXd 
  EvaluateEquationAt(const Eigen::ArrayXXd &x) = 0;

  /**
   * @brief Evaluate the Equation and get its derivatives
   * 
   * Evaluation of the Equation along points x and the graident
   * of the equation with respect to x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this Equation 
   * along the points x and the derivative of the equation with respect to x.
   */
  virtual EvalAndDerivative
  EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x) = 0;

  /**
   * @brief Evaluate the Equation and get its derivatives.
   * 
   * Evaluation of the this Equation along the points x and the gradient
   * of the equation with respect to the constants of the equation.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this Equation 
   * along the points x and the derivative of the equation with respect to 
   * the constants of the equation.
   */
  virtual EvalAndDerivative
  EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x) = 0;

  /**
   * @brief Get the Complexity of this Equation.
   * 
   * @return int 
   */
  virtual int GetComplexity() = 0;

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_