from ..EHCRegressor import complexity, model, est
from ellyn import ellyn


est.op_list = ['n','v','+','-','*','/','exp','log','2','3','sqrt','sin','cos']
# double the evals
est.eHC_its=5
est.g = 143
est.popsize = 1000
est.time_limit = 8*60*60
