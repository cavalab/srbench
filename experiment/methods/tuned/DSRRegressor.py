from ..DSRRegressor import complexity,model,base_config
from dsr import DeepSymbolicRegressor
import numpy as np
import copy
from .params._dsrregressor import params

#double the evals
base_config['training']['n_samples'] = 1000000

# Create the model
est = DeepSymbolicRegressor(base_config)

