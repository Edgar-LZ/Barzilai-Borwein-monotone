from Modules.params import get_params, _define_function_params
from Modules.position_track import optimize_method_position_track
from Modules.functions import functions_class
from numpy.random import normal
from numpy import array

methods = [
    "steepest",
    "barzilai",
    "ANGM",
    "ANGR1",
    "ANGR2",
]

params = get_params()
params["function name"] = "quadratic"
params["tau"] = 1e-6
x = normal(0, 0.5, (2))
for method in methods:
    params["method"] = method
    params = _define_function_params(params)
    function = functions_class(params)
    params["x"] = x.copy()
    algorithm = optimize_method_position_track(params)
    algorithm.run()
