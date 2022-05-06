from numpy import array, dot, inf
from typing import Callable


class get_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        if params["search name"] == "bisection":
            self.method = self.bisection
        if params["search name"] == "back tracking":
            self.method = self.back_tracking
        if params["search name"] == "barzilai":
            self.method = self.barzilai

    def bisection(self, function: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        # Inicialización
        x, gradient_i = information
        x_i, x = x
        alpha = 0.0
        beta_i = inf
        alpha_k = 1
        dot_grad = gradient(x, params) @ d
        while True:
            armijo_condition = self.obtain_armijo_condition(function,
                                                            dot_grad,
                                                            x,
                                                            params,
                                                            d,
                                                            alpha_k)
            wolfe_condition = self.obtain_wolfe_condition(gradient,
                                                          x,
                                                          params,
                                                          dot_grad,
                                                          d,
                                                          alpha_k)
            if armijo_condition or wolfe_condition:
                if armijo_condition:
                    beta_i = alpha_k
                    alpha_k = 0.5*(alpha + beta_i)
                else:
                    alpha = alpha_k
                    if beta_i == inf:
                        alpha_k = 2.0 * alpha
                    else:
                        alpha_k = 0.5 * (alpha + beta_i)
            else:
                break
        return alpha_k

    def back_tracking(self, function: Callable, gradient: Callable, information: array, params: dict, d: array):
        """
        Calcula tamaño de paso alpha

            Parámetros
            -----------
                x_k     : Vector de valores [x_1, x_2, ..., x_n]
                d_k     : Dirección de descenso
                f       : Función f(x)
                f_grad  : Función que calcula gradiente
                alpha   : Tamaño inicial de paso
                ro      : Ponderación de actualización
                c1      : Condición de Armijo
            Regresa
            -----------
                alpha_k : Tamaño actualizado de paso
        """
        # Inicialización
        alpha_k = self.params["alpha bisection"]
        x, gradient_i = information
        x_i, x = x
        dot_grad = -gradient(x, params) @ d
        # Repetir hasta que se cumpla la condición de armijo
        while True:
            armijo_condition = self.obtain_armijo_condition(function,
                                                            dot_grad,
                                                            x,
                                                            params,
                                                            d,
                                                            alpha_k)
            if armijo_condition:
                alpha_k = self.params["rho"] * alpha_k
            else:
                break
        return alpha_k

    def obtain_armijo_condition(self, function: Callable, dot_grad: float, x: array, params: dict, d: array, alpha: float):
        """
        Condicion de armijo
        """
        params_copy = params.copy()
        fx_alphagrad = function(params)
        params_copy["x"] = params_copy["x"]+alpha*d
        fx_alpha = function(params_copy)
        fx_alphagrad += self.params["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self, gradient: Callable,  x: array, params: dict, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = gradient(x+alpha*d, params)
        dfx_alpha = dfx_alpha @ d
        wolfe_condition = dfx_alpha < self.params["c2"]*dot_grad
        return wolfe_condition

    def barzilai(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list = information
        x_i, x_j = x_list
        gradient_i, gradient_j = gradient_list
        s_k = x_j-x_i
        y_k = gradient_j-gradient_i
        up = dot(s_k, s_k)
        down = dot(s_k, y_k)
        # down = dot(y_k, y_k)
        alpha = up/down
        return alpha
