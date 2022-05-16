/*!
 * \file utils.h
 *
 * \author Tyler Townsend
 * \author Ethan Adams
 * \date
 *
 * This file contains utility functions for doing and testing
 * sybolic regression problems in the bingo package
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

#ifndef BINGOCPP_INCLUDE_BINGOCPP_UTILS_H_
#define BINGOCPP_INCLUDE_BINGOCPP_UTILS_H_

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

namespace bingo {

typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> InputAndDeriviative;

/*! \brief Calculate derivatves with respect to time (first dimension)
 *
 *   \param[in] x array in which derivatives will be calculated in the
 *                first dimension. Distinct trajectories can be specified
 *                by separating the datasets within x by rows of nan
 *                Eigen::ArrayXXd
 *   \return std::vector<Eigen::ArrayXXd> with x array and corresponding time
 *                                        derivatives
 */
InputAndDeriviative CalculatePartials(const Eigen::ArrayXXd &x);
/*! \brief Generalized factorial
 *
 *   \param[in] a double
 *   \param[in] b double
 *   \return double factorial
 */
double GenFact(double a, double b);
/*! \brief Calculates the Gram Polynomial (gp_s=0) or its gp_s'th derivative
 *         evaluated at gp_i, order gp_k, over 2gp_m+1 points
 *
 *   \param[in] gp_i double
 *   \param[in] gp_m double
 *   \param[in] gp_k double
 *   \param[in] gp_s double
 *   \return double polynomial
 */
double GramPoly(double eval_point,
                double num_points,
                double polynomial_order,
                double derivative_order);
/*! \brief Calculates the weight of the gw_i'th data point for the gw_t'th
 *         Least-Square point of the gw_s'th derivative over 2gw_m+1 points,
 *         order gw_n
 *
 *   \param[in] gw_i double
 *   \param[in] gw_t double
 *   \param[in] gw_m double
 *   \param[in] gw_n double
 *   \param[in] gw_s double
 *   \return double weight
 */
double GramWeight(double eval_point_start,
                  double eval_point_end,
                  double num_points,
                  double ploynomial_order,
                  double derivative_order);
/*! \brief Smooth (and optionally differentiate) data with a Savitzky-Golay filter
 *    The Savitzky-Golay filter removes high frequency noise from data.
 *    It has the advantage of preserving the original shape and
 *    features of the signal better than other types of filtering
 *    approaches, such as moving averages techniques.
 *
 *    The Savitzky-Golay is a type of low-pass filter, particularly
 *    suited for smoothing noisy data. The main idea behind this
 *    approach is to make for each point a least-square fit with a
 *    polynomial of high order over a odd-sized window centered at
 *    the point.
 *
 *    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
 *       Data by Simplified Least Squares Procedures. Analytical
 *       Chemistry, 1964, 36 (8), pp 1627-1639.
 *    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
 *       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
 *       Cambridge University Press ISBN-13: 9780521880688
 *
 *  \param[in] y array_like, shape (N,) the values of the time history
 *               of the signal. Eigen::ArrayXXd
 *  \param[in] window_size the length of the window. Must be an odd integer
 *                         number. int
 *  \param[in] order the order of the polynomial used in the filtering. Must
 *                   be less than window_size - 1. int
 *  \param[in] deriv the order of the derivative to compute (default = 0 means
 *                   only smoothing). int
 *  \return Eiggen::ArrayXXd the smoothed signal (or it's n-th derivative).
 */
Eigen::ArrayXXd SavitzkyGolay(Eigen::ArrayXXd y,
                              int window_size,
                              int polynomial_order,
                              int derivative_order = 0);
} // namespace bingo 
#endif // BINGOCPP_INCLUDE_BINGOCPP_UTILS_H_
