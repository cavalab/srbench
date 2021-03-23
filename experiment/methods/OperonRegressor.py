from operon.sklearn import SymbolicRegressor
import operon._operon as op


est = SymbolicRegressor(
            allowed_symbols='add,sub,mul,div,constant,variable',
            offspring_generator='basic',
            local_iterations=0, 
            n_threads=1, 
            random_state=None, 
            )

hyper_params = {} #TODO

#TODO
def complexity(est):
    return est._model.Length

def model(est, X):
    #TODO: replace with est._model_str_ when PR merged
    return str(op.InfixFormatter.Format(est._model, op.Dataset(X), 3))
model = None
