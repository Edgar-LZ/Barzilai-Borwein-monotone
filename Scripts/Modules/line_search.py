from numpy import array, dot, eye, inf, sqrt
from numpy.linalg import norm, solve
from typing import Callable


class get_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        if params["search name"] == "fixed":
            self.method = self.fixed
        if params["search name"] == "bisection":
            self.method = self.bisection
        if params["search name"] == "barzilai":
            self.method = self.barzilai
        if params["search name"] == "ANGM":
            pass
        if params["search name"] == "ANGR1":
            pass
        if params["search name"] == "ANGR2":
            pass

    def fixed(self, function: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        # Inicialización
        alpha = 0.01
        return alpha

    def bisection(self, function: Callable, gradient: Callable, x_k: array, params: dict, d: array) -> float:
        # Inicialización
        alpha = 0.0
        beta_i = inf
        alpha_k = 1
        dot_grad = gradient(x_k, params) @ d
        while True:
            armijo_condition = self._armijo_condition(function,
                                                      dot_grad,
                                                      x_k,
                                                      params,
                                                      d,
                                                      alpha_k)
            wolfe_condition = self._wolfe_condition(gradient,
                                                    x_k,
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

    def _armijo_condition(self, function: Callable, dot_grad: float, x: array, params: dict, d: array, alpha: float):
        """
        Condicion de armijo
        """
        params_copy = params.copy()
        fx_alphagrad = function(params)
        params_copy["x"] = params_copy["x"]+alpha*d
        fx_alpha = function(params_copy)
        fx_alphagrad += self.params["c1"]*alpha*dot_grad
        armijo = fx_alpha > fx_alphagrad
        return armijo

    def _wolfe_condition(self, gradient: Callable,  x: array, params: dict, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = gradient(x+alpha*d, params)
        dfx_alpha = dfx_alpha @ d
        wolfe = dfx_alpha < self.params["c2"]*dot_grad
        return wolfe

    def barzilai(self, function_f: Callable, gradient: Callable, information: array, params: dict, d: array) -> float:
        x_list, gradient_list, hessian_list, alpha_list = information
        x, x_i, x_j = x_list
        x, gradient_i, gradient_j = gradient_list
        s_k = x_j-x_i
        # delta = norm(s_k)/norm(gradient_j)
        delta = 0.1
        y_k = gradient_j-gradient_i
        if params["BB type"] == 1:
            alpha_i = self._get_alpha_bb1(s_k,
                                          y_k)
        if params["BB type"] == 2:
            alpha_i = self._get_alpha_bb2(s_k,
                                          y_k)
        alpha = min((alpha_i, delta))
        return alpha

    def _get_alpha_bb1(self, s_k: array, y_k: array) -> float:
        """
        Ecuacion 5a
        """
        up = dot(s_k, s_k)
        down = dot(s_k, y_k)
        alpha = up/down
        return alpha

    def _get_alpha_bb2(self, s_k: array, y_k: array) -> float:
        """
        Ecuacion 5b
        """
        up = dot(s_k, y_k)
        down = dot(y_k, y_k)
        alpha = up/down
        return alpha

    def _get_alpha_sd(self, g_k: array, h_k: array) -> float:
        """
        alpha para el descenso de gradiente estandar
        Pagina 2a
        """
        alpha = dot(g_k, g_k) / (g_k@h_k@g_k)
        return alpha

    def _get_alpha_mg(self, g_k: array, h_k: array) -> float:
        """
        alpha para minimal gradient
        Ecuacion 24a
        """
        alpha = (g_k@h_k@g_k)
        alpha = alpha / (g_k@h_k@h_k@g_k)
        return alpha

    def _get_q(self, g_k: array, g_j: array) -> array:
        """
        Retorna la aproximación a q como se define en el paper
        """
        # qk = solve(eye(g_j.shape[0])-alpha_j*h_k, g_j)
        zeros = g_k == 0
        g_k[zeros] = 1
        qk = g_j**2 / g_k
        qk[zeros] = 0
        return qk

    def _get_alpha_k(self, qk, H_k):
        """
        alpha gorrito para la obtencion de BB2
        Ecuacion 24a
        """
        return self._get_alpha_mg(qk, H_k)

    def _get_gamma_k(self, q_j: array, g_k: array, H_k: array) -> float:
        """
        gamma usado para calcular BB2
        ecuacion 25b
        """
        # up_left = dot(q_j, H_k)
        # up_right = dot(H_k, g_k)
        # up = dot(up_left, up_right)
        # gamma = 4 * (up)**2
        # down_left = dot(up_left, q_j)
        # down_right = dot(g_k, up_right)
        # gamma = gamma / (down_left * down_right)
        gamma = 4 * (q_j@H_k@H_k@g_k)**2
        gamma = gamma / (q_j@H_k@q_j * g_k@H_k@g_k)
        return gamma

    def _get_alpha_bb1_paper(self, q_j: array, g_k: array, H_k: array) -> float:
        """
        Retorna el nuevo cálculo para BB1 propuesto
        """
        alpha_sd = 1/self._get_alpha_sd(g_k, H_k)
        qAq = q_j@H_k@q_j
        qk_norm = dot(q_j, q_j)
        gk_norm = dot(g_k, g_k)
        qAg = q_j@H_k@g_k
        raiz = (qAq/qk_norm - alpha_sd)**2 + 4*qAg / (qk_norm * gk_norm)
        den = qAq/qk_norm + alpha_sd + sqrt(raiz)
        alpha = 2/den
        return alpha

    def _get_alpha_bb2_paper(self, q_j: array, h_k: array, g_k: array, alpha_k_prev: array, alpha_mg: array) -> float:
        """
        alpha para la aproximacion de BB2
        Ecuacion 24
        """
        alpha_mg = 1 / alpha_mg
        alpha_k_prev = 1 / alpha_k_prev
        gamma_k = self._get_gamma_k(q_j, g_k, h_k)
        raiz = (alpha_k_prev - alpha_mg)**2 + gamma_k
        den = alpha_k_prev + alpha_mg + sqrt(raiz)
        alpha = 2/den
        return alpha
