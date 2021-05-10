from sklearn import kernel_ridge

hyper_params = [{
    'kernel': ('linear', 'poly','rbf','sigmoid',),
    'alpha': (1e-4,1e-2,0.1,1,),
    'gamma': (0.01,0.1,1,10,),
}]

est=kernel_ridge.KernelRidge()

def complexity(est):
    # this is a loose lower bound on the model complexity, basically jsut
    # capturing the number of parameters in the model. Could make this 
    # kernel-specific and more accurate.
    return est.dual_coef_.size 

model = None
eval_kwargs = {
               'scale_y': False,
               'scale_x': True,
              }
