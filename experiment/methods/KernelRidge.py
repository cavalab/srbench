from sklearn import kernel_ridge

hyper_params = [{
    'kernel': ('linear', 'poly','rbf','sigmoid',),
    'alpha': (1e-4,1e-2,0.1,1,),
    'gamma': (0.01,0.1,1,10,),
}]

est=kernel_ridge.KernelRidge()


