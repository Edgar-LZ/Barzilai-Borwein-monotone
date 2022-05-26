from Modules.params import get_methods_names, get_params, _define_function_params
from Modules.position_track import optimize_method_position_track
from Modules.functions import functions_class
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pandas import read_csv
from os.path import join
from numpy import array
from os import makedirs


def create_temporal_params(x: array) -> dict:
    params = {"x": x, "lambda": 10}
    return params


datasets = {
    "rosembrock": {
        "x ticks": range(0, 105, 5),
        "x lim": 100,
        "y1 lim": 20,
        "y2 lim": 100,
    },
    "quadratic": {
        "x ticks": range(0, 22, 2),
        "x lim": 20,
        "y1 lim": 2,
        "y2 lim": 10,
        "lambda": 10,
    },
}
methods = get_methods_names()
params = get_params()
for function_name in datasets:
    params["function name"] = function_name
    dataset = datasets[params["function name"]]
    params.update(dataset)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
    for method in methods:
        params["method"] = method
        color = methods[method]["color"]
        title = methods[method]["title"]
        params = _define_function_params(params)
        function = functions_class(params)
        algorithm = optimize_method_position_track(params)
        folder = algorithm.create_folder_results()
        filename = "{}.csv".format(params["search algorithm"])
        filename = join(folder, filename)
        data = read_csv(filename)
        positions = list(zip(data["0"], data["1"]))
        data["x"] = positions
        data["x"] = data["x"].apply(lambda x: array(x))
        data["function"] = data["x"].apply(
            lambda x: function.f(create_temporal_params(x)))
        data["gradient"] = data["x"].apply(
            lambda x: norm(function.gradient(x, create_temporal_params(x))))
        ax1.plot(data["function"],
                 marker="o",
                 ls="--",
                 alpha=0.7,
                 label=title,
                 color=color,
                 lw=3)
        ax1.grid(ls="--",
                 color="#000000",
                 alpha=0.4)
        ax1.set_xlim(0,
                     dataset["x lim"])
        ax1.set_xticks([value
                        for value in dataset["x ticks"]])
        ax1.set_ylabel("$f(x)$", fontsize=13)
        ax1.set_ylim(0,
                     dataset["y1 lim"])
        ax2.plot(data["gradient"],
                 marker=".",
                 ls="--",
                 color=color,
                 alpha=0.7,
                 lw=3)
        ax2.grid(ls="--",
                 color="#000000",
                 alpha=0.4)
        ax2.set_xlabel("Iteraciones",
                       fontsize=13)
        ax2.set_ylabel("$\\nabla f(x)$",
                       fontsize=13)
        ax2.set_ylim(0, dataset["y2 lim"])
    fig.legend(ncol=5,
               fontsize=13,
               frameon=False,
               loc="upper center")
    plt.tight_layout(pad=2.5)
    filename = "{}.png".format(params["function name"])
    folder = join(params["path graphics"],
                  "function")
    makedirs(folder,
             exist_ok=True)
    filename = join(folder,
                    filename)
    plt.savefig(filename,
                dpi=400)
