from numpy import array, dot, inf
from numpy.linalg import norm
from typing import Callable


class get_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        if params["search name"] == "bisection":
            self.method = self.bisection
        if params["search name"] == "barzilai":
            self.method = self.barzilai_stabilized
        if params["search name"] == "ANGM":
            self.method = self.angm
        if params["search name"] == "ANGR1":
            self.method = self.angr1
        if params["search name"] == "ANGR2":
            self.method = self.angr2

    def bisection(self, function: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        # Inicialización
        x, gradient_i, hessian = information
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

    def barzilai_stabilized(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list, hessian_list = information
        x_i, x_j = x_list
        gradient_i, gradient_j = gradient_list
        s_k = x_j-x_i
        # delta = norm(s_k)/norm(gradient_j)
        delta = 0.1
        y_k = gradient_j-gradient_i
        if params["BB type"] == 1:
            up = dot(s_k, s_k)
            down = dot(s_k, y_k)
        if params["BB type"] == 2:
            up = dot(s_k, y_k)
            down = dot(y_k, y_k)
        alpha = min((up/down, delta))
        return alpha

    def angm(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list, hessian_list = information
        x_i, x_j = x_list
        gradient_i, gradient_j = gradient_list
        hessian_i, hessian_j = hessian_list

    def angr1(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list, hessian_list = information
        x_i, x_j = x_list
        gradient_i, gradient_j = gradient_list
        hessian_i, hessian_j = hessian_list

    def angr2(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list, hessian_list = information
        x_i, x_j = x_list
        gradient_i, gradient_j = gradient_list
        hessian_i, hessian_j = hessian_list
