from pytest import param
from Modules.params import get_params
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join

params = get_params()
params["method"] = "ANGR1"
filename = "{}.csv".format(params["method"])
filename = join(params["path results"],
                params["gamma folder"],
                filename)
data = read_csv(filename)
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(12, 4))
ax1.scatter(data.index,
            data["Gamma"])
ax2.scatter(data.index,
            data["Max index"])
fig.text(0.49, 0.01, "Iteraciones")
plt.tight_layout()
plt.savefig("test.png")
