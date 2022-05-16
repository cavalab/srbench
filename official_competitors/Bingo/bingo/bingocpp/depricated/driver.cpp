/*!
 * \file driver.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the main function for bingocpp.
 * 
 * Notices
 * -------
 * Copyright 2018 United States Government as represented by the Administrator of 
 * the National Aeronautics and Space Administration. No copyright is claimed in 
 * the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *  
 * 
 * Disclaimers
 * -----------
 * No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
 * ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
 * TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
 * IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
 * FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
 * FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
 * SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
 * ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
 * RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
 * RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
 * DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF 
 * PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 * 
 * Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
 * UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
 * PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
 * LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
 * INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE 
 * OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED 
 * STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR 
 * RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY 
 * SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <chrono>

#include <Eigen/Dense>

#include "bingocpp/version.h"
#include "bingocpp/backend.h"


int test_eig() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}

void TestAcyclicGraph(int num_loops, int num_evals) {
  Eigen::ArrayX3i stack(12, 3);
  Eigen::ArrayX3i simple_stack(8, 3);
  Eigen::ArrayXXd x(60, 3);
  Eigen::VectorXd constants(2);
  // y = x_0 * ( C_0 + C_1/x_1 ) - x_0
  stack << 0, 0, 0,
        0, 1, 1,
        1, 0, 0,
        1, 1, 1,
        5, 3, 1,
        5, 3, 1,
        2, 4, 2,
        2, 4, 2,
        4, 6, 0,
        4, 5, 6,
        3, 7, 6,
        3, 8, 0;
  simple_stack << 0, 0, 0,
               0, 1, 1,
               1, 0, 0,
               1, 1, 1,
               5, 3, 1,
               2, 4, 2,
               4, 5, 0,
               3, 6, 0;
  x << 1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
  1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.;
  constants << 3.14, 10.0;
  //PrintStack(stack);
  Eigen::ArrayXXd y;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy;
  double avg_time_per_eval = 0.;
  double avg_time_per_seval = 0.;
  double avg_time_per_deval = 0.;
  double avg_time_per_sdeval = 0.;

  for (int i = 0; i < num_loops; ++i) {
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double, std::micro> duration;
    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_evals; ++i) {
      y = bingo::backend::Evaluate(stack, x, constants);
    }

    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_eval += duration.count() / num_evals;
    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_evals; ++i) {
      y = bingo::backend::Evaluate(simple_stack, x, constants);
    }

    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_seval += duration.count() / num_evals;
    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_evals; ++i) {
      y_and_dy = bingo::backend::EvaluateWithDerivative(stack, x, constants);
    }

    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_deval += duration.count() / num_evals;
    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_evals; ++i) {
      y_and_dy = bingo::backend::EvaluateWithDerivative(simple_stack, x, constants);
    }

    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_sdeval += duration.count() / num_evals;
  }

  avg_time_per_eval /= num_loops;
  avg_time_per_seval /= num_loops;
  avg_time_per_deval /= num_loops;
  avg_time_per_sdeval /= num_loops;
  std::cout << "Eval:              " << avg_time_per_eval << " microseconds\n";
  std::cout << "Simple Eval:       " << avg_time_per_seval << " microseconds\n";
  std::cout << "Eval Deriv:        " << avg_time_per_deval << " microseconds\n";
  std::cout << "Simple Eval Deriv: " << avg_time_per_sdeval << " microseconds\n";
}


int main(int argc, char *argv[]) {
  srand (time(NULL));

  if (argc < 3) {
    fprintf(stdout, "%s Version %d.%d.%d.%d\n", argv[0], PROJECT_VERSION_MAJOR,
            PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH, PROJECT_VERSION_TWEAK);
    fprintf(stdout, "Usage: %s number number\n", argv[0]);
    return 1;
  }

  TestAcyclicGraph(std::atol(argv[1]), std::atol(argv[2]));
  return 0;
}






