from operon.sklearn import SymbolicRegressor

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
    return None
