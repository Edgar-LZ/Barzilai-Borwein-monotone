from pandas import DataFrame, read_csv
from pytest import param
from Modules.params import get_function_names, get_search_methods_names, get_params
from os import listdir as ls
from os.path import join

params = get_params()
function_names = get_function_names()
methods = get_search_methods_names()
index_subresults = ["Function",
                    "Gradient",
                    "Iterations"]
function_columns = sorted(function_names*len(index_subresults))
results_columns = index_subresults*len(function_names)
mean_results = DataFrame(columns=[function_columns,
                                  results_columns],
                         index=methods)
deviation_results = mean_results.copy()
for function_name in function_names:
    params["function name"] = function_name
    for method in methods:
        folder = join(params["path results"],
                      function_name,
                      method)
        files = sorted(ls(folder))
        subresults = DataFrame(index=files,
                               columns=index_subresults)
        for file in files:
            filename = join(folder,
                            file)
            data = read_csv(filename)
            n = len(data)
            data = data.loc[n-1]
            data = data.T
            data["Iterations"] = n
            subresults.loc[file] = data
        mean = subresults.mean()
        deviation = subresults.std()
        for column in index_subresults:
            mean_results.loc[method][(function_name, column)] = mean[column]
            deviation_results.loc[method][(
                function_name, column)] = deviation[column]
filename = join(params["path results"],
                "mean.csv")
mean_results.index.name = "Metodos"
mean_results.to_csv(filename)
filename = join(params["path results"],
                "deviation.csv")
deviation_results.to_csv(filename)
