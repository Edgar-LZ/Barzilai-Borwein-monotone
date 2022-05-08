from Modules.orchestra_problem import orchestra_problem
from Modules.params import get_params

params = get_params()
problem = orchestra_problem()
for i in range(100):
    print("Iteración {:>4}".format(i+1))
    filename = str(i).zfill(3)
    filename = "{}.csv".format(filename)
    problem.init(params)
    problem.solve()
    problem.save_results(filename)
