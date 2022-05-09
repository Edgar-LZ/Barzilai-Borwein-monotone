from Modules.params import get_params, _define_function_params
from Modules.position_track import optimize_method_position_track
from Modules.functions import functions_class
from numpy import array


params = get_params()
params["function name"] = "quadratic"
params["method"] = "ANGR2"
params["tau"] = 1e-6
params = _define_function_params(params)
function = functions_class(params)
params["x"] = array([-1, 1])
algorithm = optimize_method_position_track(params)
algorithm.run()
