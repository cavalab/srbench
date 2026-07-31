from bsr.bsr_class import BSR

hyper_params = []
for val, itrNum in zip([50,100,250],[250,100,50]):
    for treeNum in [2,3]:
        hyper_params.append(
                    {'treeNum': [treeNum], 
                     'itrNum': [itrNum], 
                     'val': [val],
                    })
# initialize
est = BSR(
          val=100,
          itrNum=500,
          treeNum=3,
          alpha1= 0.4, 
          alpha2= 0.4, 
          beta= -1, 
          disp=False, 
          max_time=60*60)

def complexity(est):
    return est.complexity()

def model(est):
    model_str = est.model()

    # get rid of square brackets
    new_model_str = model_str.replace('[','').replace(']','')
    # BSR uses caret notation for powers; SymPy parses ^ as XOR here.
    new_model_str = new_model_str.replace('^', '**')
    
    return new_model_str
