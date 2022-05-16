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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_STRING_GENERATION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_STRING_GENERATION_H_

#include <unordered_map>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Core>


#include <bingocpp/agraph/operator_definitions.h>

typedef std::unordered_map<int, std::string> PrintMap;

namespace bingo {
namespace string_generation {

const PrintMap kStackPrintMap {
  {Op::kAddition, "({}) + ({})"},
  {Op::kSubtraction, "({}) - ({})"},
  {Op::kMultiplication, "({}) * ({})"},
  {Op::kDivision, "({}) / ({}) "},
  {Op::kSin, "sin ({})"},
  {Op::kCos, "cos ({})"},
  {Op::kExponential, "exp ({})"},
  {Op::kLogarithm, "log ({})"},
  {Op::kPower, "({}) ^ ({})"},
  {Op::kAbs, "abs ({})"},
  {Op::kSqrt, "sqrt ({})"},
  {Op::kSafePower, "(|{}|) ^ ({})"},
  {Op::kSinh, "sinh ({})"},
  {Op::kCosh, "cosh ({})"},
};

const PrintMap kLatexPrintMap {
  {Op::kAddition, "{} + {}"},
  {Op::kSubtraction, "{} - ({})"},
  {Op::kMultiplication, "({})({})"},
  {Op::kDivision, "\\frac{ {} }{ {} }"},
  {Op::kSin, "sin{ {} }"},
  {Op::kCos, "cos{ {} }"},
  {Op::kExponential, "exp{ {} }"},
  {Op::kLogarithm, "log{ {} }"},
  {Op::kPower, "({})^{ ({}) }"},
  {Op::kSafePower, "(|{}|)^{ ({}) }"},
  {Op::kAbs, "|{}|"},
  {Op::kSqrt, "\\sqrt{ {} }"},
  {Op::kSinh, "sinh{ {} }"},
  {Op::kCosh, "cosh{ {} }"},
};

const PrintMap kConsolePrintMap {
  {Op::kAddition, "{} + {}"},
  {Op::kSubtraction, "{} - ({})"},
  {Op::kMultiplication, "({})({})"},
  {Op::kDivision, "({})/({})"},
  {Op::kSin, "sin({})"},
  {Op::kCos, "cos({})"},
  {Op::kExponential, "exp({})"},
  {Op::kLogarithm, "log({})"},
  {Op::kPower, "({})^({})"},
  {Op::kSafePower, "(|{}|)^({})"},
  {Op::kAbs, "|{}|"},
  {Op::kSqrt, "sqrt({})"},
  {Op::kSinh, "sinh({})"},
  {Op::kCosh, "cosh({})"},
};

 /**
   * @brief Get a formatted string for an agraph
   *
   * Formatting an Agraph into a string.
   *
   * @param format A string specifying the desired format.
   *
   * @param command_array The stack specifying the commands of the Agraph.
   *
   * @param constants The constants contained in the equation.
   *
   * @return std::string The formatted string.
   */
std::string GetFormattedString(const std::string format,
                               const Eigen::ArrayX3i &command_array,
                               const Eigen::VectorXd &constants);

} // namespace string_generation
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_STRING_GENERATION_H_
