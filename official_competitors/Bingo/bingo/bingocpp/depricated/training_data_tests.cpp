/*!
 * \file training_data.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for TrainingData class
 */

#include <iostream>

#include "gtest/gtest.h"
#include "bingocpp/explicit_regression.h"
#include "bingocpp/implicit_regression.h"
#include "bingocpp/training_data.h"
#include "bingocpp/utils.h"
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace bingo;

TEST(TrainingDataTest, ExplicitConstruct) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd y(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  y << 6, 7, 1, 2, 4, 5, 8, 9;
  ExplicitTrainingData ex = ExplicitTrainingData(x, y);

  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLE_EQ(x(i, 0), ex.x(i, 0));
    ASSERT_DOUBLE_EQ(x(i, 1), ex.x(i, 1));
    ASSERT_DOUBLE_EQ(x(i, 2), ex.x(i, 2));
    ASSERT_DOUBLE_EQ(y(i, 0), ex.y(i, 0));
    ASSERT_DOUBLE_EQ(y(i, 1), ex.y(i, 1));
  }
}

TEST(TrainingDataTest, ExplicitGetItem) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd y(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  y << 6, 7, 1, 2, 4, 5, 8, 9;
  std::vector<int> items;
  items.push_back(1);
  items.push_back(3);
  ExplicitTrainingData ex = ExplicitTrainingData(x, y);
  ExplicitTrainingData* slice = ex.GetItem(items);
  Eigen::ArrayXXd truth_x(2, 3);
  Eigen::ArrayXXd truth_y(2, 2);
  truth_x << 4, 5, 6, 7, 4, 7;
  truth_y << 1, 2, 8, 9;

  for (int i = 0; i < 2; ++i) {
    ASSERT_DOUBLE_EQ(truth_x(i, 0), slice->x(i, 0));
    ASSERT_DOUBLE_EQ(truth_x(i, 1), slice->x(i, 1));
    ASSERT_DOUBLE_EQ(truth_x(i, 2), slice->x(i, 2));
    ASSERT_DOUBLE_EQ(truth_y(i, 0), slice->y(i, 0));
    ASSERT_DOUBLE_EQ(truth_y(i, 1), slice->y(i, 1));
  }
}

TEST(TrainingDataTest, ExplicitSize) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd y(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  y << 6, 7, 1, 2, 4, 5, 8, 9;
  ExplicitTrainingData ex = ExplicitTrainingData(x, y);
  ASSERT_EQ(4, ex.Size());
}

TEST(TrainingDataTest, ImplicitConstruct) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd dx_dt(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;
  ImplicitTrainingData im = ImplicitTrainingData(x, dx_dt);

  for (int i = 0; i < 4; ++i) {
    ASSERT_DOUBLE_EQ(x(i, 0), im.x(i, 0));
    ASSERT_DOUBLE_EQ(x(i, 1), im.x(i, 1));
    ASSERT_DOUBLE_EQ(x(i, 2), im.x(i, 2));
    ASSERT_DOUBLE_EQ(dx_dt(i, 0), im.dx_dt(i, 0));
    ASSERT_DOUBLE_EQ(dx_dt(i, 1), im.dx_dt(i, 1));
  }
}

TEST(TrainingDataTest, ImplicitGetItem) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd dx_dt(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;
  std::vector<int> items;
  items.push_back(1);
  items.push_back(3);
  ImplicitTrainingData im = ImplicitTrainingData(x, dx_dt);
  ImplicitTrainingData* slice = im.GetItem(items);
  Eigen::ArrayXXd truth_x(2, 3);
  Eigen::ArrayXXd truth_dx_dt(2, 2);
  truth_x << 4, 5, 6, 7, 4, 7;
  truth_dx_dt << 1, 2, 8, 9;

  for (int i = 0; i < 2; ++i) {
    ASSERT_DOUBLE_EQ(truth_x(i, 0), slice->x(i, 0));
    ASSERT_DOUBLE_EQ(truth_x(i, 1), slice->x(i, 1));
    ASSERT_DOUBLE_EQ(truth_x(i, 2), slice->x(i, 2));
    ASSERT_DOUBLE_EQ(truth_dx_dt(i, 0), slice->dx_dt(i, 0));
    ASSERT_DOUBLE_EQ(truth_dx_dt(i, 1), slice->dx_dt(i, 1));
  }
}

TEST(TrainingDataTest, ImplicitSize) {
  Eigen::ArrayXXd x(4, 3);
  Eigen::ArrayXXd dx_dt(4, 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
  dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;
  ImplicitTrainingData im(x, dx_dt);
  ASSERT_EQ(4, im.Size());
}

TEST(UtilsTest, savitzky_golay) {
  Eigen::ArrayXXd y(9, 2);
  y << 7, 4, 3, 11, 2, 13, 6, 15, 10, 22, 0, 14, 18, 19, 2, 15, 13, 8;
  Eigen::ArrayXXd truth(9, 1);
  truth << -.0595238, -1.0119, -.964286, .0833333, 2.96032, -.18254, .186508,
        1.72222, 4.4246;
  Eigen::ArrayXXd sav = savitzky_golay(y, 7, 3, 1);

  for (int i = 0; i < 9; ++i) {
    ASSERT_NEAR(sav(i), truth(i), .01);
  }
}

TEST(UtilsTest, calculate_partials) {
  Eigen::ArrayXXd x(8, 3);
  x << 1., 4., 7., 2., 5., 8., 3., 6., 9.,
  5., 1., 4., 5., 6., 7., 8., 4., 5.,
  7., 3., 14., 5.64, 8.28, 11.42;
  Eigen::ArrayXXd x_truth(1, 3);
  Eigen::ArrayXXd time_deriv_truth(1, 3);
  x_truth << 5, 1, 4;
  time_deriv_truth << 1.53175, -.178571, -1.86905;
  InputAndDeriviative cal = CalculatePartials(x);

  for (int i = 0; i < 1; ++i) {
    ASSERT_NEAR(cal.first(i, 0), x_truth(i, 0), .001);
    ASSERT_NEAR(cal.first(i, 1), x_truth(i, 1), .001);
    ASSERT_NEAR(cal.second(i, 0), time_deriv_truth(i, 0), .001);
    ASSERT_NEAR(cal.second(i, 1), time_deriv_truth(i, 1), .001);
  }
}