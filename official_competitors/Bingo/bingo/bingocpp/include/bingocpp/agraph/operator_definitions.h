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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
#define BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
#include <string>
#include <vector>
#include <unordered_map>

namespace bingo {

enum Op : signed int {
  kInteger=-1,
  kVariable=0,
  kConstant=1,
  kAddition=2,
  kSubtraction=3,
  kMultiplication=4,
  kDivision=5,
  kSin=6,
  kCos=7,
  kExponential=8,
  kLogarithm=9,
  kPower=10,
  kAbs=11,
  kSqrt=12,
  kSafePower=13,
  kSinh=14,
  kCosh=15,
};

const std::unordered_map<int, bool>  kIsArity2Map = {
  {Op::kInteger, false},
  {Op::kVariable, false},
  {Op::kConstant, false},
  {Op::kAddition, true},
  {Op::kSubtraction, true},
  {Op::kMultiplication, true},
  {Op::kDivision, true},
  {Op::kSin, false},
  {Op::kCos, false},
  {Op::kExponential, false},
  {Op::kLogarithm, false},
  {Op::kPower, true},
  {Op::kSafePower, true},
  {Op::kAbs, false},
  {Op::kSqrt, false},
  {Op::kSinh, false},
  {Op::kCosh, false},
};

const std::unordered_map<int, bool> kIsTerminalMap = {
  {Op::kInteger, true},
  {Op::kVariable, true},
  {Op::kConstant, true},
  {Op::kAddition, false},
  {Op::kSubtraction, false},
  {Op::kMultiplication, false},
  {Op::kDivision, false},
  {Op::kSin, false},
  {Op::kCos, false},
  {Op::kExponential, false},
  {Op::kLogarithm, false},
  {Op::kPower, false},
  {Op::kSafePower, false},
  {Op::kAbs, false},
  {Op::kSqrt, false},
  {Op::kSinh, false},
  {Op::kCosh, false},
};

const std::unordered_map<int, std::vector<std::string>> kOperatorNames {
  {Op::kVariable, std::vector<std::string> {"integer"}},
  {Op::kVariable, std::vector<std::string> {"load", "x"}},
  {Op::kConstant, std::vector<std::string> {"constant", "c"}},
  {Op::kAddition, std::vector<std::string> {"add", "addition", "+"}},
  {Op::kSubtraction, std::vector<std::string> {"subtract", "subtraction", "-"}},
  {Op::kMultiplication, std::vector<std::string> {"multiply", "multiplication", "*"}},
  {Op::kDivision, std::vector<std::string> {"divide", "division", "/"}},
  {Op::kSin, std::vector<std::string> {"sine", "sin"}},
  {Op::kCos, std::vector<std::string> {"cosine", "cos"}},
  {Op::kExponential, std::vector<std::string> {"exponential", "exp", "e"}},
  {Op::kLogarithm, std::vector<std::string> {"logarithm", "log"}},
  {Op::kPower, std::vector<std::string> {"power", "pow", "^"}},
  {Op::kAbs, std::vector<std::string> {"absolute value", "||", "|"}},
  {Op::kSqrt, std::vector<std::string> {"square root", "sqrt"}},
  {Op::kSafePower, std::vector<std::string> {"safe power", "safe pow"}},
  {Op::kSinh, std::vector<std::string> {"sineh", "sinh"}},
  {Op::kCosh, std::vector<std::string> {"cosineh", "cosh"}},
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
