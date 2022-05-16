import sys
import time
import numpy as np
from build import bingocpp


def TestAcyclicGraph(num_loops, num_evals):
    x = np.array(
      [1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
      dtype=float)
    x = x.reshape([60,3]);
    stack = np.array([(0, 0, 0), 
             (0, 1, 1), 
             (1, 0, 0), 
             (1, 1, 1), 
             (5, 3, 1),
             (5, 3, 1),
             (2, 4, 2),
             (2, 4, 2),
             (4, 6, 0),
             (4, 5, 6),
             (3, 7, 6),
             (3, 8, 0)], dtype=int)
    # stack = bingocpp.CommandStack(pystk)
    consts = np.array([3.14, 10.])
    
    avg_time_per_eval = 0.0
    avg_time_per_seval = 0.0
    avg_time_per_deval = 0.0
    avg_time_per_sdeval = 0.0
    for _ in range(num_loops):
        t0 = time.time()
        for _ in range(num_evals): 
            y = bingocpp.evaluate_equation_at(stack, x, consts)
        t1 = time.time()
        avg_time_per_eval += (t1 -t0)/num_evals
        
        t0 = time.time()
        for _ in range(num_evals): 
            y = bingocpp.simplify_and_evaluate(stack, x, consts)
        t1 = time.time()
        avg_time_per_seval += (t1 -t0)/num_evals
        
        t0 = time.time()
        for _ in range(num_evals): 
            y = bingocpp.evaluate_with_derivative(stack, x, consts)
        t1 = time.time()
        avg_time_per_deval += (t1 -t0)/num_evals
        
        t0 = time.time()
        for _ in range(num_evals): 
            y = bingocpp.simplify_and_evaluate_with_derivative(stack, x, consts)
        t1 = time.time()
        avg_time_per_sdeval += (t1 -t0)/num_evals
    avg_time_per_eval /= num_loops
    avg_time_per_seval /= num_loops
    avg_time_per_deval /= num_loops
    avg_time_per_sdeval /= num_loops
    print("Evaluate:              ", 
          avg_time_per_eval*1e6, " microseconds")
    print("Simple Evaluate:       ", 
          avg_time_per_seval*1e6, " microseconds")
    print("Evaluate Deriv:        ", 
          avg_time_per_deval*1e6, " microseconds")
    print("Simple Evaluate Deriv: ", 
          avg_time_per_sdeval*1e6, " microseconds")



if __name__=='__main__':
  TestAcyclicGraph(int(sys.argv[1]), int(sys.argv[2]))

