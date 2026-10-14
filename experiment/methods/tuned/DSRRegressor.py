from dsr import DeepSymbolicRegressor

from ..DSRRegressor import base_config
from ..DSRRegressor import complexity as complexity
from ..DSRRegressor import model as model

# double the evals
base_config["training"]["n_samples"] = 1000000

# Create the model
est = DeepSymbolicRegressor(base_config)
