from taylorGP.genetic import SymbolicRegressor


from sklearn.metrics import mean_squared_error  # 均方误差
import pandas as pd
import numpy as np
import time
import random

est = SymbolicRegressor(population_size=10, init_depth=(2, 5),
                        generations=1000, stopping_criteria=1e-10,
                        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt'],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.2,
                        max_samples=1.0, verbose=0,
                        parsimony_coefficient=0.1,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), low_memory=True)

def model(est, X=None):
    '''
    Return a sympy-compatible string of the final model.

    Parameters
    ----------
    X: pd.DataFrame, default=None
        The training data existing variables and targets. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model.
    '''
    '''
    mapping = {'x' + str(i): k for i, k in enumerate(X.columns)}
    new_model = est._program
    for k, v in mapping.items():
        new_model = str(new_model).replace(k, v)
    return new_model
    '''
    return est.sympy_program
def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X
       versus relative to FEAT's batch size setting.
    """
    if len(X)>1000:
        est.generations = 500
    print('TaylorGP gens adjusted to',est.generations)

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {'generations': 5,
                                  'population_size': 10
                                 }
                  )
if __name__ == '__main__':
    '''
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileName', default='example.tsv', type=str)
    args = argparser.parse_args()    
    '''
    # X_Y_name= np.loadtxt("example.tsv",dtype=np.str)[:1]
    # print(X_Y_name[0][0])
    # X_Y = np.loadtxt("example.tsv", dtype=np.float, skiprows=1)
    #输入改为pandas格式
    # X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(0) + ".tsv",dtype=np.float, skiprows=1)
    X_Y = np.loadtxt(r"example.tsv", dtype=np.float, skiprows=1)
    # hsplit函数可以水平分隔数组，该函数有两个参数，第 1 个参数表示待分隔的数组， 第 2 个参数表示要将数组水平分隔成几个小数组
    # X,Y=np.hsplit(X_Y,2)
    np.random.shuffle(X_Y)
    X, Y = np.split(X_Y, (-1,), axis=1)
    _split = int(X.shape[0] * 0.75)
    train_X = X[:_split]
    train_y = Y[:_split]

    test_X = X[_split:]
    test_y = Y[_split:]

    # X_Y = pd.read_csv("example.tsv",sep=' ',header=None)
    # print("="*1000)
    # X, Y = np.split(X_Y, (-1,), axis=1)
    est.fit(train_X,train_y.reshape(-1))#先加上测试，后面再删掉
    fitness = mean_squared_error(est.predict(test_X), test_y, squared=False)  # RMSE
    print('rmse_fitness: ', fitness)
    # print("model=",model(est,X))



