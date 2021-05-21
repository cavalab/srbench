from bsr.bsr_class import BSR

hyper_params = []
for val, itrNum in zip([100,500,1000],[5000,1000,500]):
    for treeNum in [3,6]:
        hyper_params.append(
                    {'treeNum': [treeNum], 
                     'itrNum': [itrNum], 
                     'val': [val],
                    })
# initialize
est = BSR(
          val=100,
          itrNum=5000,
          treeNum=3,
          alpha1= 0.4, 
          alpha2= 0.4, 
          beta= -1, 
          disp=False, 
          max_time=2*60*60)

def complexity(est):
    return est.complexity()

def model(est):
    return est.model()
