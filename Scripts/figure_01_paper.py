from Modules.params import get_params, _define_function_params
from Modules.functions import functions_class
from Modules.methods import algorithm_class
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import array, dot
from typing import Callable


def barzilai(function_class: Callable, algorithm: Callable, params: dict) -> tuple:
    """
        Metodo del descenso del gradiente con paso de barzilai
        """
    # Inicializacion del vector de resultado
    function = function_class.f
    gradient = function_class.gradient
    iteration = 1
    x_j = params["x"].copy()
    x_k = params["x"].copy()
    gradient_k = gradient(x_k, params)
    gamma_list = []
    max_index_list = []
    while(True):
        alpha = params["alpha"]
        if iteration == 1:
            pass
        else:
            s_k = x_k-x_j
            y_k = gradient_k-gradient_j
            if params["BB type"] == 1:
                alpha = algorithm.get_alpha._get_alpha_bb1(s_k, y_k)
            if params["BB type"] == 2:
                alpha = algorithm.get_alpha._get_alpha_bb2(s_k, y_k)
            alpha = min((alpha, 0.1))
        # Guardado del paso anterior
        x_j = x_k.copy()
        # Siguiente paso
        x_k = x_k - alpha*gradient_k
        gradient_j = gradient_k.copy()
        gamma, max_index = obtain_gamma(gradient_j)
        gamma_list += [gamma]
        max_index_list += [max_index[0]]
        # Calculo del gradiente en el paso i
        gradient_k = gradient(x_k, params)
        params["x"] = x_k
        if algorithm.stop_functions.gradient(gradient,
                                             x_k,
                                             params):
            break
        if algorithm.stop_functions.iterations(iteration,
                                               params):
            break
        iteration += 1
    return array(gamma_list), array(max_index_list)


def angm(function_class: Callable, algorithm: Callable, params: dict) -> tuple:
    """
        Metodo del descenso del gradiente con paso de barzilai
        """
    # Inicializacion del vector de resultado
    function = function_class.f
    gradient = function_class.gradient
    hessian = function_class.hessian
    tau_1 = params["tau 1"]
    tau_2 = params["tau 2"]
    x_k = params["x"].copy()
    iteration = 1
    gradient_k = gradient(x_k, params)
    hessian_k = hessian(x_k, params)
    # Inicializacion de variables
    q_k = array([])
    alpha_mg = algorithm.get_alpha._get_alpha_mg(gradient_k,
                                                 hessian_k)
    alpha_k_bb2 = None
    alpha_k = None
    gamma_list = []
    max_index_list = []
    while(True):
        alpha = params["alpha"]
        if iteration >= 2:
            q_j = q_k.copy()
            q_k = algorithm.get_alpha._get_q(gradient_k,
                                             gradient_j)
            s_k = x_k-x_j
            y_k = gradient_k-gradient_j
            alpha_k_bb1 = algorithm.get_alpha._get_alpha_bb1(s_k,
                                                             y_k)
            alpha_j_bb2 = alpha_k_bb2
            alpha_k_bb2 = algorithm.get_alpha._get_alpha_bb2(s_k,
                                                             y_k)
            alpha_j = alpha_k
            alpha_k = algorithm.get_alpha._get_alpha_k(q_k,
                                                       hessian_k)
            alpha = alpha_k_bb1
        if iteration >= 3:
            alpha_bb2_paper = algorithm.get_alpha._get_alpha_bb2_paper(q_j,
                                                                       hessian_k,
                                                                       gradient_k,
                                                                       alpha_j,
                                                                       alpha_mg)
            decision_1 = alpha_k_bb2 < tau_1*alpha_k_bb1
            decision_2 = norm(gradient_j) < tau_2*norm(gradient_k)
            if decision_1 and decision_2:
                alpha = min(alpha_k_bb2, alpha_j_bb2)
            elif decision_1 and not decision_2:
                alpha = alpha_bb2_paper
            else:
                alpha = alpha_k_bb1
        x_j = x_k.copy()
        x_k = x_k-alpha*gradient_k
        gradient_j = gradient_k.copy()
        gamma, max_index = obtain_gamma(gradient_j)
        gamma_list += [gamma]
        max_index_list += [max_index[0]]
        gradient_k = gradient(x_k, params)
        hessian_k = hessian(x_k, params)
        alpha_mg = algorithm.get_alpha._get_alpha_mg(gradient_k,
                                                     hessian_k)
        params["x"] = x_k
        if algorithm.stop_functions.gradient(gradient,
                                             x_k,
                                             algorithm.params):
            break
        if algorithm.stop_functions.iterations(iteration,
                                               algorithm.params):
            break
        iteration += 1
    return array(gamma_list), array(max_index_list)


def angr1(function_class: Callable, algorithm: Callable, params: dict) -> tuple:
    """
        Metodo del descenso del gradiente con paso de barzilai
        """
    # Inicializacion del vector de resultado
    function = function_class.f
    gradient = function_class.gradient
    hessian = function_class.hessian
    tau_1 = params["tau 1"]
    tau_2 = params["tau 2"]
    x_k = params["x"].copy()
    iteration = 1
    gradient_k = gradient(x_k, params)
    hessian_k = hessian(x_k, params)
    # Inicializacion de variables
    q_k = array([])
    q_j = array([])
    alpha_k_bb2 = None
    alpha_k = None
    alpha_j = None
    gamma_list = []
    max_index_list = []
    while(True):
        alpha = params["alpha"]
        if iteration >= 2:
            q_i = q_j.copy()
            q_j = q_k.copy()
            q_k = algorithm.get_alpha._get_q(gradient_k,
                                             gradient_j)
            s_k = x_k-x_j
            y_k = gradient_k-gradient_j
            alpha_k_bb1 = algorithm.get_alpha._get_alpha_bb1(s_k,
                                                             y_k)
            alpha_j_bb2 = alpha_k_bb2
            alpha_k_bb2 = algorithm.get_alpha._get_alpha_bb2(s_k,
                                                             y_k)
            alpha_i = alpha_j
            alpha_j = alpha_k
            alpha_k = algorithm.get_alpha._get_alpha_k(q_k,
                                                       hessian_k)
            alpha = alpha_k_bb1
        if iteration >= 4:
            alpha_bb2_paper = algorithm.get_alpha._get_alpha_bb2_paper(q_i,
                                                                       hessian_j,
                                                                       gradient_j,
                                                                       alpha_i,
                                                                       alpha_k_bb2)
            decision_1 = alpha_k_bb2 < tau_1*alpha_k_bb1
            decision_2 = norm(gradient_j) < tau_2*norm(gradient_k)
            if decision_1 and decision_2:
                alpha = min(alpha_k_bb2, alpha_j_bb2)
            elif decision_1 and not decision_2:
                alpha = alpha_bb2_paper
            else:
                alpha = alpha_k_bb1
        x_j = x_k.copy()
        x_k = x_k-alpha*gradient_k
        gradient_j = gradient_k.copy()
        gamma, max_index = obtain_gamma(gradient_j)
        gamma_list += [gamma]
        max_index_list += [max_index[0]]
        gradient_k = gradient(x_k, params)
        hessian_j = hessian_k.copy()
        hessian_k = hessian(x_k, params)
        params["x"] = x_k
        if algorithm.stop_functions.gradient(gradient,
                                             x_k,
                                             params):
            break
        if algorithm.stop_functions.iterations(iteration,
                                               params):
            break
        iteration += 1
    return array(gamma_list), array(max_index_list)


def angr2(function_class: Callable, algorithm: Callable, params: dict) -> tuple:
    """
    Metodo del descenso del gradiente con paso de barzilai
    """
    # Inicializacion del vector de resultado
    function = function_class.f
    gradient = function_class.gradient
    hessian = function_class.hessian
    tau_1 = params["tau 1"]
    tau_2 = params["tau 2"]
    x_k = params["x"].copy()
    iteration = 1
    gradient_k = gradient(x_k, params)
    hessian_k = hessian(x_k, params)
    # Inicializacion de variables
    q_k = array([])
    q_j = array([])
    alpha_mg = algorithm.get_alpha._get_alpha_mg(gradient_k,
                                                 hessian_k)
    alpha_k_bb2 = None
    alpha_k = None
    alpha_j = None
    alpha_k_true = None
    gamma_list = []
    max_index_list = []
    while(True):
        alpha_j_true = alpha_k_true
        alpha_k_true = params["alpha"]
        if iteration >= 2:
            q_j = q_k.copy()
            q_k = algorithm.get_alpha._get_q(gradient_k,
                                             gradient_j)
            s_k = x_k-x_j
            y_k = gradient_k-gradient_j
            alpha_k_bb1 = algorithm.get_alpha._get_alpha_bb1(s_k,
                                                             y_k)
            alpha_j_bb2 = alpha_k_bb2
            alpha_k_bb2 = algorithm.get_alpha._get_alpha_bb2(s_k,
                                                             y_k)
            alpha_i = alpha_j
            alpha_j = alpha_k
            diff = q_k-gradient_j
            alpha_k = alpha_j_true*dot(q_k, diff)
            alpha_k = alpha_k / dot(diff, diff)
            alpha = alpha_k_bb1
        if iteration >= 4:
            decision_1 = alpha_k_bb2 < tau_1*alpha_k_bb1
            decision_2 = norm(gradient_j) < tau_2*norm(gradient_k)
            if decision_1 and decision_2:
                alpha_k_true = min(alpha_k_bb2, alpha_j_bb2)
            elif decision_1 and not decision_2:
                alpha_k_true = min(alpha_k_bb2, alpha_i)
            else:
                alpha_k_true = alpha_k_bb1
        x_j = x_k.copy()
        x_k = x_k-alpha_k_true*gradient_k
        gradient_j = gradient_k.copy()
        gamma, max_index = obtain_gamma(gradient_j)
        gamma_list += [gamma]
        max_index_list += [max_index[0]]
        gradient_k = gradient(x_k, params)
        hessian_k = hessian(x_k, params)
        params["x"] = x_k
        if algorithm.stop_functions.gradient(gradient,
                                             x_k,
                                             params):
            break
        if algorithm.stop_functions.iterations(iteration,
                                               params):
            break
        iteration += 1
    return array(gamma_list), array(max_index_list)


def obtain_gamma(gradient: array) -> tuple:
    max_index = gradient.argsort()[-3:][::-1]
    sum_vector = sum(abs(gradient))
    values = gradient[max_index]
    sum_values = sum(abs(values))
    gamma = sum_values/sum_vector
    return gamma, max_index


params = get_params()
params["function name"] = "paper"
params["method"] = "barzilai"
params["method"] = "ANGR1"
params["tau"] = 1e-6
params = _define_function_params(params)
function = functions_class(params)
params["x"] = array([10
                     for i in range(10)])
algorithm = algorithm_class(params)
gamma, max_index = angr1(function,
                         algorithm,
                         params)
num1 = sum(gamma > 0.8)
total = len(gamma)
print("{} de {}".format(num1, total))
index = range(len(gamma))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(index, gamma)
ax2.scatter(index, max_index)
plt.show()
