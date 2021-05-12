import numpy as np
import pandas as pd

def jsonify(d):
    """recursively formats dicts for json serialization"""
    if isinstance(d, list):
        d_new = []
        for v in d:
            d_new.append(jsonify(v))
        return d_new
    elif isinstance(d, dict):
        for k in d.keys():
            d[k] = jsonify(d[k])
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif d.__class__.__name__.startswith('int'):
        return int(d)
    elif d.__class__.__name__.startswith('float'):
        return float(d)
    elif isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.values.tolist()
    elif isinstance(d, bool):
        return d
    elif d == None:
        return None
    elif not isinstance(d, str):
        print("WARNING: attempting to store ",d,"as a str for json")
        return str(d)
    return d

import sympy
from yaml import load, Loader
import pdb

def get_sym_model(dataset, return_str=True):
    """return sympy model from dataset metadata"""
    metadata = load(
            open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
            Loader=Loader
    )
    df = pd.read_csv(dataset,sep='\t')
    features = [c for c in df.columns if c != 'target']
#     print('features:',df.columns)
    description = metadata['description'].split('\n')
    model_str = [ms for ms in description if '=' in ms][0].split('=')[-1]
    if return_str:
        return model_str
#     print('model:',model_str)
    model_sym = parse_expr(model_str, 
			   local_dict = {k:Symbol(k) for k in features})
#     print('sym model:',model_sym)
    return model_sym
