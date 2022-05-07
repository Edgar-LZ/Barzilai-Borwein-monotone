from numpy import array, fill_diagonal, zeros, sum, zeros_like, log10


class functions_class:
    """
    Funciones que se usaran para ser optimizadas
    """

    def __init__(self, params: dict) -> None:
        name = params["function name"]
        if name == "wood":
            self.f = self.f_wood
            self.gradient = self.gradient_wood
            self.hessian = self.hessian_wood
        if name == "rosembrock":
            self.f = self.f_rosembrock
            self.gradient = self.gradient_rosembrock
            self.hessian = self.hessian_rosembrock
        if name == "paper":
            self.f = self.function_paper
            self.gradient = self.gradient_paper
            self.hessian = self.hessian_paper

    def f_wood(self, params: dict) -> float:
        """
        Funcion de Wood
        """
        x = params["x"]
        fx = 100*(x[0]*x[0]-x[1])*(x[0]*x[0]-x[1])
        fx += (x[0]-1)*(x[0]-1)+(x[2]-1)*(x[2]-1)
        fx += 90*(x[2]*x[2]-x[3])*(x[2]*x[2]-x[3])
        fx += 10.1*((x[1]-1)*(x[1]-1)+(x[3]-1)*(x[3]-1))
        fx += 19.8*(x[1]-1)*(x[3]-1)
        return fx

    def gradient_wood(self, x: array, params: dict) -> array:
        """
        Gradiente de Wood
        """
        n = x.shape[0]
        g = zeros(n)
        g[0] = 400*(x[0]*x[0]-x[1])*x[0] + 2*(x[0]-1)
        g[1] = -200*(x[0]*x[0]-x[1])+20.2*(x[1]-1)+19.8*(x[3]-1)
        g[2] = 2*(x[2]-1)+360*(x[2]*x[2]-x[3])*x[2]
        g[3] = -180*(x[2]*x[2]-x[3])+20.2*(x[3]-1)+19.8*(x[1]-1)
        return g

    def hessian_wood(self, x: array, params: dict) -> array:
        """ 
        Hessiano de wood
        """
        n = x.shape[0]
        h = zeros((n, n),
                  dtype=float)

        h[0][0] = 1200 * x[0]**2 - 400 * x[1] + 2
        h[0][1] = h[1][0] = -400 * x[0]
        h[1][1] = 220.2
        h[2][2] = 1080 * x[2]**2 - 360 * x[3] + 2
        h[3][1] = h[1][3] = 19.8
        h[3][2] = h[2][3] = -360 * x[2]
        h[3][3] = 200.2
        return h

    def f_rosembrock(self, params: dict) -> float:
        """
        Funcion de Rosembrock
        """
        x = params["x"]
        # x
        x_i = x[:-1]
        # x+1
        x_j = x[1:]
        fx = 100*(x_j-x_i**2)**2
        fx += (1-x_i)**2
        fx = sum(fx)
        return fx

    def gradient_rosembrock(self, x: array, params: dict) -> array:
        """
        Gradiente de Rosembrock
        """
        g = zeros_like(x)
        # x-1
        x_i = x[:-2]
        # x
        x_j = x[1:-1]
        # x+1
        x_k = x[2:]
        g[1:-1] = 200*(x_j-x_i**2)
        g[1:-1] = -400*x_j*(x_k-x_j**2)
        g[1:-1] = -2*(1-x_j)
        g[0] = -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
        g[-1] = 200*(x[-1]-x[-2]**2)
        return g

    def hessian_rosembrock(self, x: array, params: dict) -> array:
        """ 
        Hessiano de rosembrock
        """
        n = x.shape[0]
        h = zeros((n, n),
                  dtype=float)
        for i in range(n-1):
            h[i][i] = -400 * x[i+1] + 1200 * x[i]**2 + 2
            h[i][i] += 200 if i != 0 else 0
            h[i][i+1] = h[i+1][i] = -400 * x[i]
        h[n-1][n-1] = 200.0
        return h

    def function_paper(self, params: dict) -> float:
        """
        Funcion definida en el paper 
        Ecuacion 10
        """
        x = params["x"]
        matrix = self._get_matriz_paper()
        f = x@matrix@x/2
        return f

    def gradient_paper(self, x: array, params: dict) -> array:
        """
        Gradiente de la ecuación definida en el paper
        """
        matrix = self._get_matriz_paper()
        g = matrix@x
        return g

    def hessian_paper(self, x: array, params: dict) -> array:
        """
        Hessiano de la ecuacion definida en el paper
        """
        matrix = self._get_matriz_paper()
        return matrix

    def _get_matriz_paper(self) -> array:
        k = 10**3
        ncond = log10(k)
        n = 10
        a = array([10**(ncond*(n-j)/(n-1))
                   for j in range(1, n+1)])
        matrix = zeros((n, n))
        fill_diagonal(matrix, a)
        return matrix
