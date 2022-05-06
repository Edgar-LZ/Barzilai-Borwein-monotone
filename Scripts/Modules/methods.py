from .stop_criteria import stop_functions
from .functions import functions_class
from .line_search import get_alpha
from numpy.linalg import norm
from pandas import DataFrame
from numpy import array


class algorithm_class:
    """
    Contenido de los métodos para realizar una optimización
    """

    def __init__(self, params: dict) -> None:
        # Parametros de las funciones y elecciones
        self.params = params
        # Creacion del dataframe de los resultados
        self.create_results_dataframe()
        self.function = functions_class(params)
        # Funcion para obtener el alpha que cumpla las condiciones
        self.get_alpha = get_alpha(params)
        # Funciones para detener el metodo
        self.stop_functions = stop_functions(params["tau"])
        # Eleccion del metodo
        if params["search algorithm"] == "barzilai":
            self.method = self.descent_gradient_with_barzalai
        if params["search algorithm"] != "barzilai":
            self.method = self.descent_gradient

    def create_results_dataframe(self) -> DataFrame:
        self.results = DataFrame(columns=["Function",
                                          "Gradient",
                                          "Alpha"])

    def save_results(self, iteration: int, function: float, gradient: array, alpha: float) -> None:
        norm_gradient = norm(gradient)
        self.results.loc[iteration] = [function,
                                       norm_gradient,
                                       alpha]
        text = "Iteration: {:>5} "
        text += "Function: {:>25} "
        text += "Gradient {:>15}"
        print(text.format(iteration,
                          function,
                          norm_gradient))

    def descent_gradient(self):
        """
        Metodo del descenso del gradiente con paso de barzalai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        iteration = 1
        x_j = self.params["x"].copy()
        while(True):
            # Guardado del paso anterior
            x_i = x_j.copy()
            # Calculo del gradiente en el paso i
            gradient_i = -gradient(x_i, self.params)
            # Siguiente paso
            information = [[x_j, x_j],
                           [gradient_i, gradient_i]]
            alpha = self.get_alpha.method(function,
                                          gradient,
                                          information,
                                          self.params,
                                          gradient_i)
            x_j = x_i + alpha * gradient_i
            self.params["x"] = x_j
            self.save_results(iteration,
                              function(self.params),
                              gradient(x_j, self.params),
                              alpha)
            if self.stop_functions.gradient(gradient,
                                            x_j,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1

    def descent_gradient_with_barzalai(self):
        """
        Metodo del descenso del gradiente con paso de barzalai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        iteration = 1
        x_j = self.params["x"].copy()
        x_k = self.params["x"].copy()
        gradient_k = gradient(x_k, self.params)
        gradient_i = gradient_k.copy()
        while(True):
            self.alpha_method_for_barzalai(iteration)
            x_i = x_j.copy()
            if iteration != 1:
                gradient_i = gradient_j.copy()
            gradient_j = gradient_k.copy()
            # Guardado del paso anterior
            x_j = x_k.copy()
            # Calculo del gradiente en el paso i
            gradient_d = -gradient(x_j, self.params)
            # Siguiente paso
            information = [[x_i, x_j],
                           [gradient_i, gradient_j]]
            alpha = self.get_alpha.method(function,
                                          gradient,
                                          information,
                                          self.params,
                                          gradient_d)
            x_k = x_j + alpha * gradient_d
            gradient_k = gradient(x_k, self.params)
            self.params["x"] = x_k
            self.save_results(iteration,
                              function(self.params),
                              gradient(x_k, self.params),
                              alpha)
            if self.stop_functions.gradient(gradient,
                                            x_k,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1

    def alpha_method_for_barzalai(self, iteration: int) -> None:
        # Guardado del paso anterior
        if iteration == 1 or self.params["search name"] == "bisection":
            self.select_get_alpha_method("bisection")
        else:
            self.select_get_alpha_method("barzilai")

    def select_get_alpha_method(self, method_name: str) -> None:
        params_copy = self.params.copy()
        params_copy["search name"] = method_name
        self.get_alpha = get_alpha(params_copy)
