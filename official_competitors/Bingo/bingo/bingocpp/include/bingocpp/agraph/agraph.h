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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_H_
#define BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_H_

#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <bingocpp/equation.h>

typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvalAndDerivative;
typedef std::tuple<Eigen::ArrayX3i, Eigen::ArrayX3i, Eigen::VectorXd,
                   bool, double, bool, int, bool, bool> AGraphState;

namespace bingo {

/**
 * @brief Acyclic graph represetnation of an equation.
 * 
 * This class contains most of the code necessary for the representation of an
 * acyclic graph (linear stack) in symbolic regression.
 */
class AGraph : public Equation {
 public:
  AGraph(const bool use_simplification);

  AGraph(const AGraph &agraph);
  AGraph(const AGraphState &state);
  AGraph(AGraph&&) = default;                
  AGraph& operator=(const AGraph&) = default; 
  AGraph& operator=(AGraph&&) = default;     
  virtual ~AGraph() = default;
  
  /**
   * @brief Creates a copy of this AGraph
   * 
   * @return AGraph
   */
  AGraph Copy();

  /**
   * @brief Serializes the AGraph object
   *
   * @return AGraphState
   */
   AGraphState DumpState();

  /**
   * @brief Get the Command Array object
   * 
   * @return Eigen::ArrayX3i The command array for this graph.
   */
  const Eigen::ArrayX3i &GetCommandArray() const;


  Eigen::ArrayX3i &GetCommandArrayModifiable();
  
  /**
   * @brief Set the Command Array object
   * 
   * @param command_array A copy of the new Command Array
   */
  void SetCommandArray(const Eigen::ArrayX3i &command_array);

  /**
   * @brief Get the Fitness of this AGraph
   * 
   * @return double 
   */
  double GetFitness() const;

  /**
   * @brief Set the Fitness for this AGraph
   * 
   * @param fitness 
   */
  void SetFitness(double fitness);

  /**
   * @brief Check if fitness has been set for this AGraph object.
   * 
   * @return true if set
   * @return false otherwise
   */
  bool IsFitnessSet() const;
  void SetFitnessStatus(bool val);

  /**
   * @brief Set the Genetic Age of this AGraph
   * 
   * @param age The age of this AGraph
   */
  void SetGeneticAge(const int age);

  /**
   * @brief Get the Genetic Age of this AGraph
   * 
   * @return int 
   */
  int GetGeneticAge() const;

  /**
   * @brief Get the Utilized Commands for the CommandArray
   * 
   * Returns a mask of all the utilized commands in the stack for 
   * representing the AGraph. The indicies in the vector represent the
   * indicies in the CommandArray
   * 
   * @return std::vector<bool> The mask.
   */
  std::vector<bool> GetUtilizedCommands() const;

  /**
   * @brief The AGraph needs local optimization.
   * 
   * Determine if the agraph needs local optimzation.
   * 
   * @return true Needs optimization.
   * @return false Has been optimized
   */
  bool NeedsLocalOptimization();

  /**
   * @brief Get the Number Local Optimization Params
   * 
   * The number of parameters that need to be optimized in this 
   * AGraph.
   * 
   * @return int The number of parameters to optimize.
   */
  int GetNumberLocalOptimizationParams();

  /**
   * @brief Set the Local Optimization Params
   * 
   * Set the optimized constants
   * 
   * @param params The optimized constants.
   */
  void SetLocalOptimizationParams(Eigen::VectorXd params);

  /**
   * @brief Get the constants in the graph.
   * 
   * Returns the constants in the graph. The AGraph should be optimized
   * before calling this method.
   * 
   * @return Eigen::VectorXd The constants in the AGraph
   */
  const Eigen::VectorXd &GetLocalOptimizationParams() const;

  /**
   * @brief Evaluate the AGraph equatoin
   * 
   * Evaluation of the AGraph Equation at points x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return Eigen::ArrayXXd The evaluation of function at points x.
   */
  Eigen::ArrayXXd 
  EvaluateEquationAt(const Eigen::ArrayXXd &x);

  /**
   * @brief Evaluate the AGraph and get its derivatives
   * 
   * Evaluation of the AGraph equation along points x and the graident
   * of the equation with respect to x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this AGraph
   * along the points x and the derivative of the equation with respect to x.
   */
  EvalAndDerivative
  EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x);

  /**
   * @brief Evluate the AGraph and get its derivatives.
   * 
   * Evaluation of the AGraph equation along the points x and the gradient
   * of the equation with respect to the constants of the equation.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this AGraph
   * along the points x and the derivative of the equation with respect to 
   * the constants of the equation.
   */
  EvalAndDerivative
  EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x);


  /**
   * @brief Output a string description of the the AGraph in a given format.
   *
   * @param format The requested format of the equation. Options are "console",
   * "latex", and "stack".
   *
   * @param node Output of the raw command array rather than the processed
   * version. Default False.
   *
   * @return std::string Equation in specified form.
   */
  std::string GetFormattedString(std::string format, bool raw);
  std::string GetConsoleString();


  /**
   * @brief Get the Complexity of this AGraph equation.
   * 
   * @return int 
   */
  int GetComplexity();

  int Distance(const AGraph &agraph);

 private:
  Eigen::ArrayX3i command_array_;
  Eigen::ArrayX3i simplified_command_array_;
  Eigen::VectorXd simplified_constants_;
  bool needs_opt_;
  double fitness_;
  bool fit_set_;
  int genetic_age_;
  bool modified_;
  bool use_simplification_;

  // To string operator when passed into stream
  friend std::ostream &operator<<(std::ostream&, AGraph&);

  // helper functions
  void notify_agraph_modification();
  void update();

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_H_
