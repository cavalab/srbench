from taylorGP.genetic import SymbolicRegressor
from taylorGP.calTaylor import Metrics,Metrics2 # ,cal_Taylor_features
from taylorGP._program import print_program
from taylorGP._global import _init,set_value

import numpy as np
from sklearn.metrics import mean_squared_error  # 均方误差
import time
import sys
import argparse
import random
from sympy import *
_init()
set_value('TUIHUA_FLAG',False)
def Taylor_Based_SR(_x,X,Y,qualified_list,low_polynomial):
    f_low_taylor = qualified_list[-5]
    f_low_taylor_mse = qualified_list[-4]
    if low_polynomial == False:
        est = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                                   generations=100, stopping_criteria=1e-10,
                                   function_set= ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt'],
                                   p_crossover=0.7, p_subtree_mutation=0.,
                                   p_hoist_mutation=0., p_point_mutation=0.2,
                                   max_samples=1.0, verbose=1,
                                   parsimony_coefficient=0.1,
                                   n_jobs=1,  #
                                   const_range=(-1, 1),
                                   random_state=random.randint(1,100), low_memory=True)
        print(qualified_list)
        est.fit(X, Y, qualified_list)
        if est._program.raw_fitness_ > f_low_taylor_mse:
            print(f_low_taylor, f_low_taylor_mse, sep='\n')
            return f_low_taylor_mse,f_low_taylor
        else:
            return est._program.raw_fitness_,print_program(est._program.get_expression(), qualified_list, X,_x)
    else:
        return f_low_taylor_mse, f_low_taylor
def model( X_Y=None):
    '''
    Return a sympy-compatible string of the final model.

    Parameters
    ----------
    X_Y: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model.
    '''
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 ,x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols("x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29 ")
    _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19 ,x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 ]
    average_fitness = 0
    repeat = 1
    time_start1 = time.time()
    time_start2 = time.time()
    X,Y = np.split(X_Y, (-1,), axis=1)
    loopNum = 0
    Metric = []
    while True:
        metric = Metrics(varNum=X.shape[1], dataSet=X_Y)
        loopNum += 1
        Metric.append(metric)
        if loopNum == 2 and X.shape[1] <= 2:
            break
        elif loopNum == 5 and (X.shape[1]>2 and X.shape[1]<=3):
            break
        elif loopNum == 4 and (X.shape[1]>3 and X.shape[1]<=4):
            break
        elif loopNum == 3 and (X.shape[1]>4 and X.shape[1]<=5):
            break
        elif loopNum == 2 and (X.shape[1]>5 and X.shape[1]<=6):
            break
        elif loopNum == 1 and (X.shape[1]>6):
            break
    Metric.sort(key=lambda x: x.nmse)
    metric = Metric[0]
    print('NMSE of polynomial and lower order polynomial after sorting:', metric.nmse, metric.low_nmse)
    if metric.nmse < 0.1:
        metric.nihe_flag = True
    else:
        metric.bias = 0.
        print('Fitting failed')
    end_fitness,program = None,None
    if metric.judge_Low_polynomial():
        end_fitness, program = metric.low_nmse,metric.f_low_taylor
    else:
        qualified_list = []
        qualified_list.extend(
            [metric.judge_Bound(),
             metric.f_low_taylor,
             metric.low_nmse,
             metric.bias,
             metric.judge_parity(),
             metric.judge_monotonicity()])
        print(qualified_list)
        end_fitness,program = Taylor_Based_SR(_x,X,metric.change_Y(Y),qualified_list,metric.low_nmse<1e-5)
    print('fitness_and_program',end_fitness,program,sep=' ')
    average_fitness += end_fitness
    time_end2 = time.time()
    print('current_time_cost', (time_end2 - time_start2) / 3600, 'hour')

    time_end1 = time.time()
    print('average_time_cost', (time_end1 - time_start1) / 3600 / repeat, 'hour')
    print('average_fitness = ',average_fitness/repeat)
    return program
if __name__ == '__main__':
    '''
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileName', default='example.tsv', type=str)
    args = argparser.parse_args()    
    '''
    X_Y = np.loadtxt("example.tsv", dtype=np.float, skiprows=1)
    print("="*1000)
    print("model=",model(X_Y))



