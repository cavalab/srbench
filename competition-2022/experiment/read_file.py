import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_df(filename, label, sep, DataFrame, **kwargs):
    if filename.endswith('gz'):
        compression = 'gzip'
    else:
        compression = None
    
    print('compression:',compression)
    print('filename:',filename)

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
               **kwargs )
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python', **kwargs )
   
    # clean up column names
    clean_names = {k:k.strip().replace('.','_') for k in input_data.columns}
    input_data = input_data.rename(columns=clean_names)

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1) 
    if not DataFrame:
        X = X.values

    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, input_data.index.values, feature_names



def read_stage0_file(filename, DataFrame=True, label='target',sep=None,
                     random_state=None):
    X, y, idx, feature_names= load_df(filename, label, sep, DataFrame)

    X_train, X_test, y_train, y_test, idx_train, idx_test = \
            train_test_split(X, y, idx,
                             train_size=0.75, 
                             test_size=0.25,
                             random_state=random_state
                            )

    return X_train, X_test, y_train, y_test, idx_train, idx_test, feature_names


def read_stage1_file(filename, label='y', sep=',', DataFrame=True):
    trainfile = filename.split('.csv')[0]+'_train.csv'
    testfile = filename.split('.csv')[0]+'_test.csv'

    if 'seir' in filename:
        label = filename.split('seir')[-1][:2]
        print('label:',label)

    X_train, y_train, idx_train, feature_names = load_df(trainfile, label, sep, DataFrame)
    X_test, y_test, idx_test, _ = load_df(testfile, label, sep, DataFrame)

    return X_train, X_test, y_train, y_test, idx_train, idx_test, feature_names

def read_stage2_file(filename, sep=',', DataFrame=True):
    """read real world data"""
    trainfile = filename #.split('.csv')[0]+'_train.csv'
    testfile = filename.replace('train','test')

    if 'cases' in filename:
        label='value_cases'
    elif 'hosp' in filename:
        label='value_hosp'
    elif 'deaths' in filename:
        label='value_deaths'
    else:
        raise ValueError(f"label for {filename} not found")

    X_train, y_train, idx_train, feature_names = load_df(trainfile, label, sep,
            DataFrame, index_col='date')
    X_test, y_test, idx_test, _ = load_df(testfile, label, sep, DataFrame, 
            index_col='date')

    return X_train, X_test, y_train, y_test, idx_train, idx_test, feature_names

def read_file(filename, DataFrame=True, stage=0, random_state=None):
    if stage == 0:
        return read_stage0_file(filename, DataFrame=DataFrame,
                                random_state=random_state)
    elif stage == 1: 
        return read_stage1_file(filename, DataFrame=DataFrame)
    elif stage == 2: 
        return read_stage2_file(filename, DataFrame=DataFrame)
