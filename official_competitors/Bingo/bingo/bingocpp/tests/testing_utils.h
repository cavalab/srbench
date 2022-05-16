#ifndef BINGO_TEST_TESTING_UTILS_H_
#define BINGO_TEST_TESTING_UTILS_H_

#include <iostream>

#include <Eigen/Dense>

namespace testutils {
namespace {

const double TESTING_TOL = 1e-7;

inline void print_difference(const Eigen::ArrayXXd &array1,
                             const Eigen::ArrayXXd &array2) {
  std::cout << "  Actual: \n" << array1 << "\n";
  std::cout << "Expected: \n" << array2 << "\n";
}
  
inline double difference(double val_1, double val_2) {
  if ((std::isnan(val_1) && std::isnan(val_2)) ||
      (val_1 == INFINITY && val_2 == INFINITY) ||
      (val_1 == -INFINITY && val_2 == -INFINITY)) {
    return 0;
  } else {
    return val_1 - val_2;
  }
}

inline bool non_comparable_matrices(const Eigen::ArrayXXd &array1,
                                    const Eigen::ArrayXXd &array2) {
  int rows_array1 = array1.rows();
  int rows_array2 = array2.rows();
  int cols_array1 = array1.cols();
  int cols_array2 = array2.cols();
  return (!(rows_array1 > 0) || 
          !(rows_array2 > 0) || 
          !(cols_array1 > 0) || 
          !(cols_array2 > 0) || 
          (rows_array1 != rows_array2) || 
          (cols_array1 != cols_array2));
}
} // namespace

inline bool almost_equal(const Eigen::ArrayXXd &array1,
                         const Eigen::ArrayXXd &array2,
                         double tolerance = TESTING_TOL) {
  if (non_comparable_matrices(array1, array2)) {
    print_difference(array1, array2);
    return false;
  }

  int rows_array1 = array1.rows();
  int cols_array1 = array1.cols();
  Eigen::MatrixXd matrix_diff = Eigen::MatrixXd(rows_array1, cols_array1);
  for (int row = 0; row < rows_array1; row++) {
    for (int col = 0; col < cols_array1; col++) {
      matrix_diff(row, col) = difference(array1(row, col), array2(row, col));
    }
  }

  double frobenius_norm = matrix_diff.norm();
  bool equal = (frobenius_norm < tolerance ? true : false);
  if (!equal) {
    std::cout << "Frobenius Norm: " << frobenius_norm << std::endl;
    print_difference(array1, array2);
  }
  return equal;
}
} // namespace testutils
#endif //BINGO_TEST_TESTING_UTILS_H_
