
import bsr
from bsr.bsr_class import BSR


# number of simple signals
K = 3
MM = 50

hyper_params = [{'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1}]
# initialize
my_bsr = BSR(K,MM)
# train (need to fill in parameters)
my_bsr.fit()
# fit new values
fitted_y = my_bsr.predict()
# display fitted trees
express = my_bsr.model()
# complexity, including complexity of each tree & total
complexity = my_bsr.complexity()
