from ..AIFeynman import complexity as complexity
from ..AIFeynman import est as est
from ..AIFeynman import model as model
from .params._aifeynman import params

est.set_params(**params)
# 8 hours
est.max_time = 8 * 60 * 60
