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
        # print("WARNING: attempting to store ",d,"as a str for json")
        return str(d)
    return d

Competitors = [
    'Bingo',
    'E2ET',
    'HROCH', 
    'PS-Tree',
    'QLattice',
    'TaylorGP',
    'eql',
    'geneticengine',
    'gpzgd',
    'nsga-dcgp',
    'operon',
    'pysr',
    'uDSR'
]

FilteredCompetitors = [
    'Bingo',
    'E2ET',
    'PS-Tree',
    'QLattice',
    'eql',
    'geneticengine',
    'gpzgd',
    'operon',
    'pysr',
    'uDSR'
]
