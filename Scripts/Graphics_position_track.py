from Modules.params import get_params, _define_function_params
from numpy import arange, array, meshgrid, zeros
from Modules.functions import functions_class
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs


def read_data(params: dict) -> dict:
    data = {
        "steepest": {"Color": "#ef476f"},
        "barzilai": {"Color": "#ffd166"},
        "ANGM":     {"Color": "#06d6a0"},
        "ANGR1":    {"Color": "#118ab2"},
        "ANGR2":    {"Color": "#d62828"},
    }
    max_size = 0
    for method in data:
        filename = "{}.csv".format(method)
        filename = join(params["path results"],
                        params["function name"],
                        params["position folder"],
                        filename)
        file = read_csv(filename)
        size = len(file)
        data[method]["data"] = file.copy()
        data[method]["size"] = size
        if max_size < size:
            max_size = size
    data["methods"] = list(data.keys())
    data["max size"] = max_size
    return data


def plot_function(ax: plt.subplot, function: functions_class) -> None:
    x = arange(-2.0, 2.1, 0.1)
    y = arange(-2.0, 2.1, 0.1)
    z = zeros((len(x), len(y)))
    x_matrix, y_matrix = meshgrid(x, y)
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            xy = array([xi, yi])
            z[i, j] = function.f({"x": xy,
                                  "lambda": 10})
    ax.contour(x_matrix,
               y_matrix,
               z,
               cmap="inferno",
               origin="lower",
               alpha=0.5,
               levels=[0, 1, 2, 4, 6, 7, 8, 9])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)


params = get_params()
params["function name"] = "quadratic"
function = functions_class(params)
data = read_data(params)
for i in range(1, data["max size"]):
    fig, ax = plt.subplots()
    plot_function(ax, function)
    for j in range(1, i):
        for method in data["methods"]:
            pos = j
            color = data[method]["Color"]
            if pos >= data[method]["size"]-1:
                pos = data[method]["size"]-1
            pos_i = data[method]["data"].loc[pos-1].to_numpy()
            pos_j = data[method]["data"].loc[pos].to_numpy()
            vector = [[pos_i[0], pos_j[0]],
                      [pos_i[1], pos_j[1]]]
            ax.scatter(pos_i[0],
                       pos_i[1],
                       marker=".",
                       c=color)
            ax.scatter(pos_j[0],
                       pos_j[1],
                       marker=".",
                       c=color)
            ax.plot(vector[0],
                    vector[1],
                    color=color,
                    lw=2,
                    ls="--")
    plt.show()
