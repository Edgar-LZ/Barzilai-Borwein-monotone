from Modules.params import get_params
from pandas import read_csv
from os.path import join


def print_latex(filename: str) -> None:
    data = read_csv(filename,
                    index_col=0,
                    header=[0, 1])
    data = data.T
    data = data.round(6)
    print(data.to_latex())


params = get_params()
filename = join(params["path results"],
                "mean.csv")
print_latex(filename)
filename = join(params["path results"],
                "deviation.csv")
print_latex(filename)
