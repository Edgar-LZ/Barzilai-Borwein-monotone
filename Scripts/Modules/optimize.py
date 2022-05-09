from .stop_criteria import stop_functions
from .functions import functions_class
from numpy import array, dot, sqrt
from .line_search import get_alpha
from numpy.linalg import norm
from pandas import DataFrame
from cmath import isnan
from sys import exit


class optimize_method:
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
        self.results = DataFrame(columns=["Function", "Gradient", "Alpha"])

    def save_results(self, iteration: int, function: float, gradient: array,
                     alpha: float) -> None:
        norm_gradient = norm(gradient)
        self.results.loc[iteration] = [function, norm_gradient, alpha]
        # text = "Iteration: {:>5} "
        # text += "Function: {:>25} "
        # text += "Gradient {:>15}"
        # print(text.format(iteration,
        #                   function,
        #                   norm_gradient))
        if isnan(function):
            text = "nan value"
            exit(text)

    def descent_gradient(self):
        """
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        iteration = 1
        x_j = self.params["x"].copy()
        while (True):
            # Guardado del paso anterior
            x_i = x_j.copy()
            # Calculo del gradiente en el paso i
            gradient_i = -gradient(x_i, self.params)
            # Siguiente paso
            alpha = self.get_alpha.method(function,
                                          gradient,
                                          x_j,
                                          self.params,
                                          gradient_i)
            x_j = x_i + alpha * gradient_i
            self.params["x"] = x_j
            self.save_results(iteration, function(self.params),
                              gradient(x_j, self.params), alpha)
            if self.stop_functions.gradient(gradient, x_j, self.params):
                break
            if self.stop_functions.iterations(iteration, self.params):
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
        while (True):
            alpha = self.params["alpha"]
            if iteration == 1:
                pass
            else:
                s_k = x_k - x_j
                y_k = gradient_k - gradient_j
                if self.params["BB type"] == 1:
                    alpha = self.get_alpha._get_alpha_bb1(s_k, y_k)
                if self.params["BB type"] == 2:
                    alpha = self.get_alpha._get_alpha_bb2(s_k, y_k)
                alpha = min((alpha, 0.1))
            # Guardado del paso anterior
            x_j = x_k.copy()
            # Siguiente paso
            x_k = x_k - alpha * gradient_k
            gradient_j = gradient_k.copy()
            # Calculo del gradiente en el paso i
            gradient_k = gradient(x_k, self.params)
            self.params["x"] = x_k
            self.save_results(iteration,
                              function(self.params),
                              gradient(x_k,
                                       self.params),
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
        tau_1 = self.params["tau 1"]
        tau_2 = self.params["tau 2"]
        x_k = self.params["x"].copy()
        iteration = 1
        gradient_k = gradient(x_k, self.params)
        hessian_k = hessian(x_k, self.params)
        # Inicializacion de variables
        q_k = array([])
        alpha_mg = self.get_alpha._get_alpha_mg(gradient_k,
                                                hessian_k)
        alpha_k_bb2 = None
        alpha_k = None
        while (True):
            alpha = self.params["alpha"]
            if iteration >= 2:
                q_j = q_k.copy()
                q_k = self.get_alpha._get_q(gradient_k,
                                            gradient_j)
                s_k = x_k - x_j
                y_k = gradient_k - gradient_j
                alpha_k_bb1 = self.get_alpha._get_alpha_bb1(s_k, y_k)
                alpha_j_bb2 = alpha_k_bb2
                alpha_k_bb2 = self.get_alpha._get_alpha_bb2(s_k, y_k)
                alpha_j = alpha_k
                alpha_k = self.get_alpha._get_alpha_k(q_k, hessian_k)
                alpha = alpha_k_bb1
            if iteration >= 3:
                alpha_bb2_paper = self.get_alpha._get_alpha_bb2_paper(q_j,
                                                                      hessian_k,
                                                                      gradient_k,
                                                                      alpha_j,
                                                                      alpha_mg)
                decision_1 = alpha_k_bb2 < tau_1 * alpha_k_bb1
                decision_2 = norm(gradient_j) < tau_2 * norm(gradient_k)
                if decision_1 and decision_2:
                    alpha = min(alpha_k_bb2, alpha_j_bb2)
                elif decision_1 and not decision_2:
                    alpha = alpha_bb2_paper
                else:
                    alpha = alpha_k_bb1
            x_j = x_k.copy()
            x_k = x_k - alpha * gradient_k
            gradient_j = gradient_k.copy()
            gradient_k = gradient(x_k, self.params)
            hessian_k = hessian(x_k, self.params)
            alpha_mg = self.get_alpha._get_alpha_mg(gradient_k,
                                                    hessian_k)
            self.params["x"] = x_k
            self.save_results(iteration,
                              function(self.params),
                              gradient_k,
                              alpha)
            if self.stop_functions.gradient(gradient, x_k,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1

    def angr1(self) -> float:
        """
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        hessian = self.function.hessian
        tau_1 = self.params["tau 1"]
        tau_2 = self.params["tau 2"]
        x_k = self.params["x"].copy()
        iteration = 1
        gradient_k = gradient(x_k, self.params)
        hessian_k = hessian(x_k, self.params)
        # Inicializacion de variables
        q_k = array([])
        q_j = array([])
        alpha_k_bb2 = None
        alpha_k = None
        alpha_j = None
        while (True):
            alpha = self.params["alpha"]
            if iteration >= 2:
                q_i = q_j.copy()
                q_j = q_k.copy()
                q_k = self.get_alpha._get_q(gradient_k,
                                            gradient_j)
                s_k = x_k - x_j
                y_k = gradient_k - gradient_j
                alpha_k_bb1 = self.get_alpha._get_alpha_bb1(s_k, y_k)
                alpha_j_bb2 = alpha_k_bb2
                alpha_k_bb2 = self.get_alpha._get_alpha_bb2(s_k, y_k)
                alpha_i = alpha_j
                alpha_j = alpha_k
                alpha_k = self.get_alpha._get_alpha_k(q_k, hessian_k)
                alpha = alpha_k_bb1
            if iteration >= 4:
                alpha_bb2_paper = self.get_alpha._get_alpha_bb2_paper(
                    q_i, hessian_j, gradient_j, alpha_i, alpha_k_bb2)
                decision_1 = alpha_k_bb2 < tau_1 * alpha_k_bb1
                decision_2 = norm(gradient_j) < tau_2 * norm(gradient_k)
                if decision_1 and decision_2:
                    alpha = min(alpha_k_bb2, alpha_j_bb2)
                elif decision_1 and not decision_2:
                    alpha = alpha_bb2_paper
                else:
                    alpha = alpha_k_bb1
            x_j = x_k.copy()
            x_k = x_k - alpha * gradient_k
            gradient_j = gradient_k.copy()
            gradient_k = gradient(x_k, self.params)
            hessian_j = hessian_k.copy()
            hessian_k = hessian(x_k, self.params)
            self.params["x"] = x_k
            self.save_results(iteration, function(self.params),
                              gradient_k,
                              alpha)
            if self.stop_functions.gradient(gradient,
                                            x_k,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1

    def angr2(self) -> float:
        """
        Metodo del descenso del gradiente con paso de barzilai
        """
        # Inicializacion del vector de resultado
        function = self.function.f
        gradient = self.function.gradient
        hessian = self.function.hessian
        tau_1 = self.params["tau 1"]
        tau_2 = self.params["tau 2"]
        x_k = self.params["x"].copy()
        iteration = 1
        gradient_k = gradient(x_k, self.params)
        hessian_k = hessian(x_k, self.params)
        # Inicializacion de variables
        q_k = array([])
        q_j = array([])
        alpha_mg = self.get_alpha._get_alpha_mg(gradient_k, hessian_k)
        alpha_k_bb2 = None
        alpha_k = None
        alpha_j = None
        alpha_k_true = None
        while (True):
            alpha_j_true = alpha_k_true
            alpha_k_true = self.params["alpha"]
            if iteration >= 2:
                q_j = q_k.copy()
                q_k = self.get_alpha._get_q(gradient_k,
                                            gradient_j)
                s_k = x_k - x_j
                y_k = gradient_k - gradient_j
                alpha_k_bb1 = self.get_alpha._get_alpha_bb1(s_k,
                                                            y_k)
                alpha_j_bb2 = alpha_k_bb2
                alpha_k_bb2 = self.get_alpha._get_alpha_bb2(s_k,
                                                            y_k)
                alpha_i = alpha_j
                alpha_j = alpha_k
                diff = q_k - gradient_j
                alpha_k = alpha_j_true * dot(q_k,
                                             diff)
                alpha_k = alpha_k / dot(diff,
                                        diff)
                alpha = alpha_k_bb1
            if iteration >= 4:
                decision_1 = alpha_k_bb2 < tau_1 * alpha_k_bb1
                decision_2 = norm(gradient_j) < tau_2 * norm(gradient_k)
                if decision_1 and decision_2:
                    alpha_k_true = min(alpha_k_bb2,
                                       alpha_j_bb2)
                elif decision_1 and not decision_2:
                    alpha_k_true = min(alpha_k_bb2,
                                       alpha_i)
                else:
                    alpha_k_true = alpha_k_bb1
            x_j = x_k.copy()
            x_k = x_k - alpha_k_true * gradient_k
            gradient_j = gradient_k.copy()
            gradient_k = gradient(x_k, self.params)
            hessian_k = hessian(x_k, self.params)
            self.params["x"] = x_k
            self.save_results(iteration,
                              function(self.params),
                              gradient_k,
                              alpha_k)
            if self.stop_functions.gradient(gradient,
                                            x_k,
                                            self.params):
                break
            if self.stop_functions.iterations(iteration,
                                              self.params):
                break
            iteration += 1
