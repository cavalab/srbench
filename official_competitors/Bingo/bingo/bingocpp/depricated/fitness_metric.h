/*!
 * \file fitness_metric.h
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the cpp version of FitnessMetric.py
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

#ifndef INCLUDE_BINGOCPP_FITNESS_METRIC_H_
#define INCLUDE_BINGOCPP_FITNESS_METRIC_H_

#include "bingocpp/acyclic_graph.h"
#include "bingocpp/training_data.h"
#include <Eigen/Dense>
#include <Eigen/Core>

namespace bingo {

struct FitnessMetric;

/*! \struct LMFunctor
 *
 *  Used for Levenberg-Marquardt Optimization
 *
 *  \fn int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec)
 *  \fn int df(const Eigen::VectorXd &x, Eigen::MatrixXf &fjac)
 *  \fn int values() const
 *  \fn int inputs() const
 */
struct LMFunctor {
  //! int m
  /*! Number of data points, i.e. values */
  int m;
  //! int n
  /*! Number of parameters, i.e. inputs */
  int n;
  //! AcyclicGraph indv
  /*! The Agraph individual */
  AcyclicGraph agraphIndv;
  //! TrainingData* train
  /*! object that holds data needed */
  TrainingData* train;
  //! FitnessMetric* fit
  /*! object that holds fitness metric */
  FitnessMetric* fit;
  /*! \brief Compute 'm' errors, one for each data point, for the given paramter values in 'x'
   *
   *  \param[in] x contains current estimates for parameters. Eigen::VectorXd (dimensions nx1)
   *  \param[in] fvec contain error for each data point. Eigen::VectorXd (dimensions mx1)
   *  \return 0
   */
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec);
  /*! \brief Compute jacobian of the errors
   *
   *  \param[in] x contains current estimates for parameters. Eigen::VectorXd (dimensions nx1)
   *  \param[in] fjac contain jacobian of the errors, calculated numerically. Eigen::MatrixXf (dimensions mxn)
   *  \return 0
   */
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac);
  /*! \brief gets the values
   *
   *  \return m - values
   */
  int values() const {
    return m;
  }
  /*! \brief gets the inputs
   *
   *  \return n - inputs
   */
  int inputs() const {
    return n;
  }

};

/*! \struct FitnessMetric
 *
 *  An abstract struct to evaluate metric based on type of regression
 *
 *  \note FitnessMetric includes : StandardRegression
 *
 *  \fn virtual Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv, TrainingData &train) = 0
 *  \fn float evaluate_fitness(AcyclicGraph &indv, TrainingData &train)
 *  \fn void optimize_constants(AcyclicGraph &indv, TrainingData &train)
 */
struct FitnessMetric {
 public:
  FitnessMetric() { }
  /*! \brief f(x) - y where f is defined by indv and x, y are in train
  *
  *  \note Each implementation will need to hard code casting TrainingData
  *        to a specific type in this function.
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData to evaluate the fitness. TrainingData
  *  \return Eigen::ArrayXXd the fitness vector
  */
  virtual Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv,
      TrainingData &train) = 0;
  /*! \brief Finds the fitness metric
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData to evaluate the fitness. TrainingData
  *  \return float the fitness metric
  */
  double evaluate_fitness(AcyclicGraph &indv, TrainingData &train);
  /*! \brief perform levenberg-marquardt optimization on embedded constants
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData used by fitness metric. TrainingData
  */
  void optimize_constants(AcyclicGraph &indv, TrainingData &train);
};
} // namespace bingo
#endif
