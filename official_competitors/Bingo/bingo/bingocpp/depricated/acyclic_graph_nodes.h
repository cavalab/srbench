/*!
 * \file acyclic_graph_nodes.hh
 *
 * \author Ethan Adams
 * \date 2/9/2018
 *
 * This is the header file to hold the Operation abstract class
 * and all implementations of that class. Also holds the OperatorInterface
 * class, which includes a map to keep the Operations in
 *
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

#ifndef INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
#define INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_

#include <vector>

#include <Eigen/Dense>

namespace agraphnodes{
  typedef void (
    *forward_operator_function)(
      const Eigen::ArrayX3i&, const Eigen::ArrayXXd&,
      const Eigen::VectorXd&, std::vector<Eigen::ArrayXXd>&, std::size_t
  );
  typedef void (
    *derivative_operator_function)(
      const Eigen::ArrayX3i &, const int,
      const std::vector<Eigen::ArrayXXd> &,
      std::vector<Eigen::ArrayXXd> &, int
  );
  
  void forward_eval_function(int node, const Eigen::ArrayX3i &stack,
                             const Eigen::ArrayXXd &x,
                             const Eigen::VectorXd &constants,
                             std::vector<Eigen::ArrayXXd> &buffer,
                             std::size_t result_location);

  void derivative_eval_function(int node, const Eigen::ArrayX3i &stack,
                                const int command_index,
                                const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                int dependency);
} //agraphnodes

#endif  // INCLUDE_BINGOCPP_ACYCLIC_GRAPH_NODES_H_
