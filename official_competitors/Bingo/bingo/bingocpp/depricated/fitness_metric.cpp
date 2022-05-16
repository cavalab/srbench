/*!
 * \file fitness_metric.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the cpp version of FitnessMetric.py
 */

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <unsupported/Eigen/NonLinearOptimization>

#include "bingocpp/explicit_regression.h"
#include "bingocpp/fitness_metric.h"

namespace bingo {
  
int LMFunctor::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) {
  agraphIndv.set_constants(x);
  fvec = fit->evaluate_fitness_vector(agraphIndv, *train);
  return 0;
}

int LMFunctor::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) {
  double epsilon;
  epsilon = 1e-5f;

  for (int i = 0; i < x.size(); i++) {
    Eigen::VectorXd xPlus(x);
    xPlus(i) += epsilon;
    Eigen::VectorXd xMinus(x);
    xMinus(i) -= epsilon;
    Eigen::VectorXd fvecPlus(values());
    operator()(xPlus, fvecPlus);
    Eigen::VectorXd fvecMinus(values());
    operator()(xMinus, fvecMinus);
    Eigen::VectorXd fvecDiff(values());
    fvecDiff = (fvecPlus - fvecMinus) / (2.0 * epsilon);
    fjac.block(0, i, values(), 1) = fvecDiff;
  }

  return 0;
}

double FitnessMetric::evaluate_fitness(AcyclicGraph &indv,
                                       TrainingData &train) {
  if (indv.needs_optimization()) {
    optimize_constants(indv, train);
  }

  return ((evaluate_fitness_vector(indv, train)).abs()).mean();
}

void FitnessMetric::optimize_constants(AcyclicGraph &indv,
                                       TrainingData &train) {
  LMFunctor functor;
  functor.train = &train;
  functor.fit = this;
  functor.m = functor.train->Size();
  // indv.input_constants();
  functor.n = indv.count_constants();
  functor.agraphIndv = indv;
  Eigen::VectorXd vec = Eigen::VectorXd::Random(functor.n);
  Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
  lm.minimize(vec);
  indv.set_constants(vec);
  indv.needs_opt = false;
}
} // namespace bingo 