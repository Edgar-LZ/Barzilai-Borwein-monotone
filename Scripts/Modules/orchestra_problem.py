from .methods import algorithm_class
from os import makedirs as mkdir
from numpy.random import normal
from numpy import array, ones
from os.path import join


class orchestra_problem:
    """
    Orquesta todos los metodos dadas una serie de parametros
    """

    def __init__(self) -> None:
        pass

    def init(self, params: dict) -> None:
        self.params = params
        self.algorithm = algorithm_class(params)
        self.initial_points()

    def initial_points(self) -> None:
        name = self.params["function name"]
        if name == "wood":
            self.params["x"] = array([-3, -1, -3, -1])
        if name == "rosembrock":
            self.params["x"] = ones(100)
            self.params["x"][0] = -1.2
            self.params["x"][-2] = -1.2
        if name == "paper":
            self.params["x"] = normal(0, 10, (10))
        if name == "quadratic":
            self.params["x"] = normal(0, 10, (2))

    def solve(self):
        """
        Orquesta la solucion del problema
        """
        # Eleccion de la funcion
        self.algorithm.method()

    def create_folder_results(self) -> str:
        folder = join(self.params["path results"],
                      self.params["function name"],
                      self.params["search name"])
        mkdir(folder,
              exist_ok=True)
        return folder

    def save_results(self, filename: str) -> None:
        folder = self.create_folder_results()
        filename = join(folder,
                        filename)
        results = self.algorithm.results
        results.index.name = "Iteration"
        results.to_csv(filename)
