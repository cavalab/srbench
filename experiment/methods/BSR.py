import bsr
from bsr.bsr_class import BSR


# number of simple signals
K = 3
MM = 50

hyper_params = [{'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1}]
# initialize
est = BSR(K,MM)

def complexity(est):
    return est.complexity()

def model(est):
    return est.model()
