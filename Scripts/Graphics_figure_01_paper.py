from Modules.params import get_methods_names, get_params, get_graphics_params
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join
from os import makedirs

params = get_params()
methods = get_methods_names()
for method in methods:
    print("-"*40)
    print("Método {}".format(method))
    params["method"] = method
    params = get_graphics_params(params)
    filename = "{}.csv".format(params["method"])
    filename = join(params["path results"],
                    params["gamma folder"],
                    filename)
    data = read_csv(filename)
    n = len(data[data["Gamma"] > 0.8])
    print("Gamma>0.8\t{} de {}".format(n, len(data)))
    # fig, (ax1, ax2) = plt.subplots(1, 2,
    #                                figsize=(12, 4),
    #                                sharex=True)
    # ax1.scatter(data.index,
    #             data["Gamma"],
    #             alpha=0.5,
    #             color="#6a040f")
    # ax1.set_xlim(params["x lim"][0],
    #              params["x lim"][1]-params["x delta"])
    # ax1.set_xticks([value
    #                 for value in range(params["x lim"][0],
    #                                    params["x lim"][1],
    #                                    params["x delta"])])
    # ax1.set_ylim(0.6, 1)
    # ax1.set_ylabel("$\\gamma (\\nabla f_x )$")
    # ax1.grid(ls="--",
    #          color="#000000",
    #          alpha=0.5)
    # ax2.scatter(data.index,
    #             data["Max index"],
    #             alpha=0.5,
    #             color="#e85d04")
    # ax2.set_yticks([value
    #                 for value in range(0, 10)])
    # ax2.set_ylim(-1, 10)
    # ax2.set_ylabel("Índice máximo de $\\nabla f_x$")
    # ax2.grid(ls="--",
    #          color="#000000",
    #          alpha=0.5)
    # fig.text(0.47, 0.01,
    #          "Iteraciones",
    #          fontsize=13)
    # filename = "{}.png".format(params["method"])
    # folder = join(params["path graphics"],
    #               params["gamma folder"])
    # makedirs(folder,
    #          exist_ok=True)
    # filename = join(folder,
    #                 filename)
    # plt.tight_layout(pad=2)
    # plt.savefig(filename,
    #             dpi=400)
