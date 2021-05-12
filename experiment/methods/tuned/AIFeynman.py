from ..AIFeynman import complexity, model, est
from .params._aifeynman import params

est.set_params(**params)
# 8 hours
est.max_time = 8*60*60

