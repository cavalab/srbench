import argparse
import os
import sys

from taylorGP.genetic import SymbolicRegressor

from sklearn.metrics import mean_squared_error  # 均方误差
import pandas as pd
from time import  time
import random
from sklearn.model_selection import train_test_split

# from pmlb import regression_dataset_name

est = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                        generations=10000, stopping_criteria=1e-10,
                        function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt'],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.2,
                        max_samples=1.0, verbose=1,
                        parsimony_coefficient=0.3,
                        n_jobs=1,  #
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), low_memory=True,
                        max_time = 3600-10)

def model(est, X):
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
    try:
        mapping = {'x' + str(i): k for i, k in enumerate(X.columns)}
        new_model = est.sympy_global_best
        for k, v in reversed(mapping.items()):
            new_model = str(new_model).replace(k, v)
        return new_model
    except BaseException:
        return est.sympy_global_best
def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X
       versus relative to FEAT's batch size setting.
    """

    if len(X)>1000:
        est.max_time = 36000 - 10 # 10 second of slack
    print('TaylorGP max_time adjusted to',est.max_time)

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {'generations': 5,
                                  'population_size': 10
                                 }
                  )
if __name__ == '__main__':
    print("hello")
    sys.setrecursionlimit(300)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--PMLBNum', default=0, type=int)
    args = argparser.parse_args()
    '''
    adult_X, adult_y = fetch_data('adult', return_X_y=True, local_cache_dir='./')
    print(adult_X)
    print(adult_y)
    print(len(dataset_names))
    print(len(classification_dataset_names))
    print("="*100)
    print(len(regression_dataset_names))
    '''
    print("="*100)

    eqName = 'time_p2_'+ str(args.PMLBNum)+'.csv'
    eq_write = open(eqName, "w+")  # 重新写
    for fileNo,regression_dataset in enumerate(regression_dataset_names[args.PMLBNum*10:args.PMLBNum*10+10]):
        print(regression_dataset)
        # X, y = fetch_data(regression_dataset, return_X_y=True,local_cache_dir='D:\PYcharm_program\pmlb\datasets')
        # local_cache_dir = "D:\PYcharm_program\pmlb\datasets"
        local_cache_dir = "/home/hebaihe/STORAGE/git_srv/pmlb/datasets"
        dataset_path = os.path.join(local_cache_dir, regression_dataset,
                                    regression_dataset+ '.tsv.gz')
        dataset = pd.read_csv(dataset_path, sep='\t', compression='gzip')
        dataset.dropna(inplace=True)
        X = dataset.drop('target', axis=1).values
        y = dataset['target'].values
        train_X, test_X, train_y, test_y = train_test_split(X, y)
        startTime = time()
        '''
        打印 X 重复的行--后面是否需要把成倍数出现的行筛掉：不需要：成倍出现不等价于采样重复
        for i in range(X.shape[0]):
            for j in range(i+1,X.shape[0]):
                if((X[i]==X[j]).all()):
                    print("data is same in :"+str(i) +" "+str(j))
        print(train_X)
        print("="*100)
        print(train_y)
        '''
        # try:
        print(train_X[:10000].shape,train_y[:10000].shape)
        if train_X.shape[1]<100:
            est.fit(train_X[:10000], train_y[:10000])
            endTime = time()
            runTime = (endTime - startTime)/60
            y_pred = est.predict(test_X[:3000])
            fitness = mean_squared_error(y_pred, test_y[:3000], squared=False)  # RMSE
            print('predict_fitness: ', fitness)
            fileIntroduce ='No.' + str(fileNo)+' '+regression_dataset
            eq_write.write( fileIntroduce + ': '
                            +'train_X: ' + str(train_X[:10000].shape)
                            +'  train cost '+str(runTime) + ' minutes  '
                            +'  train_fitness: '+str(est.global_fitness)+'  predict_fitness: '+str(fitness)
                            +'  best_program: '+str(est.sympy_global_best)+'\n')
    eq_write.close()

        # except BaseException:
        #     print(str(regression_dataset)+"Error : Singular matrix")

    '''
    # X_Y_name= np.loadtxt("example.tsv",dtype=np.str)[:1]
    # print(X_Y_name[0][0])
    # X_Y = np.loadtxt("example.tsv", dtype=np.float, skiprows=1)
    #输入改为pandas格式
    # X_Y = np.loadtxt(r"D:\PYcharm_program\Test_everything\Bench_0.15\BenchMark_" + str(0) + ".tsv",dtype=np.float, skiprows=1)
    print(simplify(sympify("x0**2+x0**4+x2**2+x1**2+x2**2+x1**2")))
    print(simplify("x0**2+x0**4+x2**2+x1**2+x2**2+x1**2"))
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
    '''




