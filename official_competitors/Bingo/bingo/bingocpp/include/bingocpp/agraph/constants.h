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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_CONSTANTS_H_
#define BINGOCPP_INCLUDE_BINGOCPP_CONSTANTS_H_

namespace bingo {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

const unsigned int kOpIdx = 0;
const unsigned int kParam1Idx = 1;
const unsigned int kParam2Idx = 2;
const unsigned int kArrayCols = 3;

const int kOptimizeConstant = -1;

} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_CONSTANTS_H_
