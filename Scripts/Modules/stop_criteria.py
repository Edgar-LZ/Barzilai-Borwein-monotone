from numpy.linalg import norm
from typing import Callable
from numpy import array


class stop_functions:
    """
    Contiene todas las funciones que son usadas para verificar si el metodo llego a un punto estacionario
    """

    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def vectors(self, vector_i: array, vector_j: array) -> bool:
        """"
        Comprueba la diferencia entre la posicion actual y la anterior
        """
        up = norm((vector_i-vector_j))
        down = max(norm(vector_j), 1)
        dot = abs(up/down)
        if dot < self.tau:
            return True
        return False

    def gradient(self, gradient: Callable, x: array, params: dict) -> bool:
        """
        Compueba si la norm del gradiente se acerca a cero
        """
        dfx = gradient(x, params)
        dfx = norm(dfx)
        if dfx < self.tau:
            return True
        return False

    def iterations(self, iteration: int, params: dict) -> bool:
        if iteration >= params["max iterations"]:
            return True
        return False
