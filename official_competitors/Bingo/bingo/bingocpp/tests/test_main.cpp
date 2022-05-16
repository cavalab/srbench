/*!
 * \file test_main.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the main function for driving unit tests.
 */

#include <math.h>

#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "gtest/gtest.h"

int main(int argc, char **argv) {
  srand (time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


