from .methods import algorithm_class
from os import makedirs as mkdir
from numpy.random import normal
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
            self.params["x"] = normal(-5, 10, (4))
        if name == "rosembrock":
            self.params["x"] = normal(-5, 2, (2))

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
