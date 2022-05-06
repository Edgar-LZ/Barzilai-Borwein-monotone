from Modules.orchestra_problem import orchestra_problem
from Modules.params import get_params

params = get_params()
problem = orchestra_problem()
problem.init(params)
problem.solve()
problem.save_results("01.csv")
