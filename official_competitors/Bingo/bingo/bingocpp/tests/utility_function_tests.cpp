/*!
 * \file utility_function_tests.cpp 
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for the utility functions 
 */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <bingocpp/utils.h>

TEST(UtilsTest, SavitzkyGolay) {
  int window_size = 7;
  int order = 3;
  int derivative_order = 1;
  Eigen::ArrayXXd y(9, 2);
  y << 7, 4,
       3, 11,
       2, 13,
       6, 15,
       10, 22,
       0, 14,
       18, 19,
       2, 15,
       13, 8;

  Eigen::ArrayXXd expected_value(9, 1);
  expected_value << -.0595238,
                    -1.0119,
                    -.964286,
                    .0833333,
                    2.96032,
                    -.18254,
                    .186508,
                    1.72222,
                    4.4246;

  Eigen::ArrayXXd computed_result = bingo::SavitzkyGolay(
        y, window_size, order, derivative_order);
  for (int i = 0; i < 9; ++i) {
    ASSERT_NEAR(computed_result(i), expected_value(i), .01);
  }
}

TEST(UtilsTest, CalculatePartials) {
  Eigen::ArrayXXd x(8, 3);
  x << 1., 4., 7.,
       2., 5., 8.,
       3., 6., 9.,
       5., 1., 4.,
       5., 6., 7.,
       8., 4., 5.,
       7., 3., 14.,
       5.64, 8.28, 11.42;
  Eigen::ArrayXXd expected_x(1, 3);
  Eigen::ArrayXXd expected_time_deriv(1, 3);
  expected_x << 5, 1, 4;
  expected_time_deriv << 1.53175, -.178571, -1.86905;
  auto partials = bingo::CalculatePartials(x);

  ASSERT_NEAR(partials.first(0, 0), expected_x(0, 0), .001);
  ASSERT_NEAR(partials.first(0, 1), expected_x(0, 1), .001);
  ASSERT_NEAR(partials.second(0, 0), expected_time_deriv(0, 0), .001);
  ASSERT_NEAR(partials.second(0, 1), expected_time_deriv(0, 1), .001);
}