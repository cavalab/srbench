/*!
 * \file acyclic_graph_tests.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the unit tests for the functions associated with the
 * acyclic graph representation of a symbolic equation.
 */

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "gtest/gtest.h"

#include "bingocpp/backend.h"
#include "test_fixtures.h"

using namespace bingo;
using namespace backend;

namespace {

class AGraphNodesTest: public::testing::Test {
 public:
  Eigen::ArrayX3i stack;
  Eigen::ArrayX3i stack2;
  Eigen::ArrayXXd x;
  Eigen::VectorXd constants;

  void SetUp() {
    stack = testutils::stack_operators_0_to_5(); 
    stack2 = testutils::stack_unary_operator(4);
    x = testutils::one_to_nine_3_by_3();
    constants = testutils::pi_ten_constants();
  }

  void TearDown() {}
};

TEST(AGraphNodesTest, XLoad) {
  Eigen::ArrayX3i stack(1, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(2);
  stack << 0, 1, 1;
  constants << 3.0, 5.0;
  x << 7., 6., 9., 5., 11., 4., 3., 2., 1.;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 6., 11., 2.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
  }
}

TEST(AGraphNodesTest, CLoad) {
  Eigen::ArrayX3i stack(1, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(2);
  stack << 1, 1, 1;
  constants << 3.0, 5.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 5., 5., 5.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
  }
}

TEST(AGraphNodesTest, Addition) {
  Eigen::ArrayX3i stack(3, 3);
  Eigen::ArrayX3i stack2(3, 3);
  Eigen::ArrayX3i stack3(2, 3);
  Eigen::ArrayX3i stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 0, 0, 0,
        1, 0, 0,
        2, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         2, 1, 0;
  stack3 << 0, 0, 0,
         2, 0, 0;
  stack4 << 1, 0, 0,
         2, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 10., 9., 12.;
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 14., 12., 18.;
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 6., 6., 6.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 1., 0., 0., 1., 0., 0., 1., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 2., 0., 0., 2., 0., 0., 2., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }


  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AGraphNodesTest, Subtraction) {
  Eigen::ArrayX3i stack(3, 3);
  Eigen::ArrayX3i stack2(3, 3);
  Eigen::ArrayX3i stack3(2, 3);
  Eigen::ArrayX3i stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 0, 0, 0,
        1, 0, 0,
        3, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         3, 1, 0;
  stack3 << 0, 2, 2,
         3, 0, 0;
  stack4 << 1, 0, 0,
         3, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 4., 3., 6.;
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << -4., -3., -6.;
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 0., 0., 0.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 1., 0., 0., 1., 0., 0., 1., 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << -1., 0., 0., -1., 0., 0., -1., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AGraphNodesTest, Multiplication) {
  Eigen::ArrayX3i stack(3, 3);
  Eigen::ArrayX3i stack2(3, 3);
  Eigen::ArrayX3i stack3(2, 3);
  Eigen::ArrayX3i stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 0, 0, 0,
        1, 0, 0,
        4, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         4, 1, 0;
  stack3 << 0, 0, 0,
         4, 0, 0;
  stack4 << 1, 0, 0,
         4, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (7. * 3.), (6. * 3.), (9. * 3.);
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << (7. * 7.), (6. * 6.), (9.* 9.);
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 9., 9., 9.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 3., 0., 0., 3., 0., 0., 3., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 14., 0., 0., 12., 0., 0., 18., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AGraphNodesTest, Division) {
  Eigen::ArrayX3i stack(3, 3);
  Eigen::ArrayX3i stack2(3, 3);
  Eigen::ArrayX3i stack3(2, 3);
  Eigen::ArrayX3i stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 0, 0, 0,
        1, 0, 0,
        5, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         5, 1, 0;
  stack3 << 0, 2, 2,
         5, 0, 0;
  stack4 << 1, 0, 0,
         5, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (7. / 3.), (6. / 3.), (9. / 3.);
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << (3. / 7.), (3. / 6.), (3. / 9.);
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 1., 1., 1.;
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 1., 1., 1.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << (1. / 3.), 0., 0., (1. / 3.), 0., 0., (1. / 3.), 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << (-3. / (pow(7., 2.))), 0., 0.,
          (-3. / (pow(6., 2.))), 0., 0.,
          (-3. / (pow(9., 2.))), 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AGraphNodesTest, Sin) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        6, 0, 0;
  stack2 << 0, 0, 0,
         6, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (sin(3.)), (sin(3.)), (sin(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (sin(7.)), (sin(6.)), (sin(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (cos(7.)), 0., 0.,
           (cos(6.)), 0., 0.,
           (cos(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AGraphNodesTest, Cos) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        7, 0, 0;
  stack2 << 0, 0, 0,
         7, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (cos(3.)), (cos(3.)), (cos(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (cos(7.)), (cos(6.)), (cos(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (-sin(7.)), 0., 0.,
           (-sin(6.)), 0., 0.,
           (-sin(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AGraphNodesTest, Exp) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        8, 0, 0;
  stack2 << 0, 0, 0,
         8, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (exp(3.)), (exp(3.)), (exp(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (exp(7.)), (exp(6.)), (exp(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (exp(7.)), 0., 0.,
           (exp(6.)), 0., 0.,
           (exp(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AGraphNodesTest, Log) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        9, 0, 0;
  stack2 << 0, 0, 0,
         9, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (log(abs(3.))), (log(abs(3.))), (log(abs(3.)));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (log(abs(7.))), (log(abs(6.))), (log(abs(9.)));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (1. / abs(7.)), 0., 0.,
           (1. / abs(6.)), 0., 0.,
           (1. / abs(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AGraphNodesTest, Power) {
  Eigen::ArrayX3i stack(3, 3);
  Eigen::ArrayX3i stack2(3, 3);
  Eigen::ArrayX3i stack3(2, 3);
  Eigen::ArrayX3i stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 0, 0, 0,
        1, 0, 0,
        10, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         10, 1, 0;
  stack3 << 0, 0, 0,
         10, 0, 0;
  stack4 << 1, 0, 0,
         10, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (pow(7., 3.)), (pow(6., 3.)), (pow(9., 3.));
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << (pow(3., 7.)), (pow(3., 6.)), (pow(3., 9.));
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << (pow(7., 7.)), (pow(6., 6.)), (pow(9., 9.));
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << (pow(3., 3.)), (pow(3., 3.)), (pow(3., 3.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << (3. * pow(7., 2.)), 0., 0.,
         (3. * pow(6., 2.)), 0., 0.,
         (3. * pow(9., 2.)), 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << (pow(3., 7.) * log(3.)), 0., 0.,
          (pow(3., 6.) * log(3.)), 0., 0.,
          (pow(3., 9.) * log(3.)), 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << (pow(7., 7.) * (1 + log(7.))), 0., 0.,
            (pow(6., 6.) * (1 + log(6.))), 0., 0.,
            (pow(9., 9.) * (1 + log(9.))), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AGraphNodesTest, Absolute) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        11, 0, 0;
  stack2 << 0, 0, 0,
         11, 0, 0;
  x << -7., 5., 3., 6., 11., 4., -9., 8., 6.;
  constants << -3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 3., 3., 3.;
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << 7., 6., 9.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << -1., 0., 0., 1., 0., 0., -1., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AGraphNodesTest, Sqrt) {
  Eigen::ArrayX3i stack(2, 3);
  Eigen::ArrayX3i stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  Eigen::VectorXd constants(1);
  stack << 1, 0, 0,
        12, 0, 0;
  stack2 << 0, 0, 0,
         12, 0, 0;
  x << -7., 5., 3., 6., 11., 4., -9., 8., 6.;
  constants << 3.0;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (sqrt(abs(3.))), (sqrt(abs(3.))), (sqrt(abs(3.)));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (sqrt(abs(-7.))), (sqrt(abs(6.))), (sqrt(abs(-9.)));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << -1. / (2.*sqrt(7.)), 0., 0.,
           1. / (2.*sqrt(6.)), 0., 0.,
           -1. / (2.*sqrt(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}
}  // namespace



