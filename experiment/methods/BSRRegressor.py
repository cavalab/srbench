from bsr.bsr_class import BSR

hyper_params = []
for treeNum, itrNum in zip([100,500,1000],[5000,1000,500]):
    for val in [10,100]:
        hyper_params.append(
                    {'treeNum': [treeNum], 
                     'itrNum': [itrNum], 
                     'val': [val],
                    })
# initialize
est = BSR( alpha1= 0.4, alpha2= 0.4, beta= -1)

def complexity(est):
    return est.complexity()

def model(est):
    return est.model()
