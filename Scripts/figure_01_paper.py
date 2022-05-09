from Modules.params import get_params, _define_function_params
from Modules.gamma_track import optimize_method_gamma_track
from Modules.functions import functions_class
import matplotlib.pyplot as plt
from numpy import array


params = get_params()
params["function name"] = "paper"
params["method"] = "barzilai"
params["tau"] = 1e-6
params = _define_function_params(params)
function = functions_class(params)
params["x"] = array([10
                     for i in range(10)])
algorithm = optimize_method_gamma_track(params)
algorithm.run()
