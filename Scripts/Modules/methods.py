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
            self.method = self.descent_gradient_with_barzilai
        if params["search algorithm"] == "ANGM":
            self.method = self.angm
        if params["search algorithm"] == "ANGR1":
            self.method = self.angr1
        if params["search algorithm"] == "ANGR2":
            self.method = self.angr2
        if params["search algorithm"] == "steepest":
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
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        iteration = 1
        x_j = self.params["x"].copy()
        hessian_dummy = 0
        alpha_dummy = 0
        while(True):
            # Guardado del paso anterior
            x_i = x_j.copy()
            # Calculo del gradiente en el paso i
            gradient_i = -gradient(x_i, self.params)
            # Siguiente paso
            information = [[x_j, x_j, x_j],
                           [gradient_i, gradient_i, gradient_i],
                           [hessian_dummy, hessian_dummy, hessian_dummy],
                           [alpha_dummy, alpha_dummy, alpha_dummy]]
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

    def descent_gradient_with_barzilai(self):
        """
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        iteration = 1
        x_j = self.params["x"].copy()
        x_k = self.params["x"].copy()
        gradient_k = gradient(x_k, self.params)
        while(True):
            alpha = self.params["alpha"]
            if iteration == 1:
                pass
            else:
                s_k = x_k-x_j
                y_k = gradient_k-gradient_j
                if self.params["BB type"] == 1:
                    alpha = self.get_alpha._get_alpha_bb1(s_k, y_k)
                if self.params["BB type"] == 2:
                    alpha = self.get_alpha._get_alpha_bb2(s_k, y_k)
                alpha = min((alpha, 0.01))
            # Guardado del paso anterior
            x_j = x_k.copy()
            # Siguiente paso
            x_k = x_k - alpha*gradient_k
            gradient_j = gradient_k.copy()
            # Calculo del gradiente en el paso i
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

    def angm(self) -> float:
        """
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        hessian = self.function.hessian
        iteration = 1
        x_l = self.params["x"].copy()
        x_k = self.params["x"].copy()
        x_j = self.params["x"].copy()
        x_i = self.params["x"].copy()
        gradient_l = gradient(x_k, self.params)
        gradient_k = gradient_l.copy()
        gradient_j = gradient_l.copy()
        gradient_i = gradient_l.copy()
        hessian_l = hessian(x_l, self.params)
        hessian_k = hessian_l.copy()
        hessian_j = hessian_k.copy()
        hessian_i = hessian_k.copy()
        alpha_l = 0.1
        alpha_k = 0.1
        alpha_j = 0.1
        alpha_i = 0.1
        while(True):
            alpha_k = alpha_l
            x_j = x_k.copy()
            x_k = x_l.copy()
            if iteration == 1:
                self.select_get_alpha_method("bisection")
            if iteration == 2:
                self.select_get_alpha_method("barzilai")
                alpha_j = alpha_k
                gradient_j = gradient_k.copy()
            if iteration > 2:
                self.select_get_alpha_method(self.params["search name"])
                x_i = x_j.copy()
                alpha_i = alpha_j
                alpha_j = alpha_k
                gradient_i = gradient_j.copy()
                gradient_j = gradient_k.copy()
            # Calculo del gradiente en el paso i
            # Guardado del paso anterior
            # x_j = x_k.copy()
            gradient_k = gradient_l.copy()
            gradient_d = -gradient(x_k, self.params)
            # Siguiente paso
            # print(gradient_i, gradient_j, gradient_k)
            information = [[x_i, x_j, x_k],
                           [gradient_i, gradient_j, gradient_k],
                           [hessian_i, hessian_j, hessian_k],
                           [alpha_i, alpha_j, alpha_k]]
            alpha = self.get_alpha.method(function,
                                          gradient,
                                          information,
                                          self.params,
                                          gradient_d)
            x_l = x_k + alpha * gradient_d
            alpha_l = alpha_k
            gradient_l = gradient(x_l, self.params)
            # hessian_j = hessian_k.copy()
            # gradient_k = gradient(x_k, self.params)
            # hessian_k = hessian(x_k, self.params)
            self.params["x"] = x_l
            self.save_results(iteration,
                              function(self.params),
                              gradient(x_l, self.params),
                              alpha)
            if self.stop_functions.gradient(gradient,
                                            x_l,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1

    def angr1(self) -> float:
        pass

    def angr2(self) -> float:
        pass
