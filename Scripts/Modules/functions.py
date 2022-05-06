from numpy import array, zeros, sum, zeros_like


class functions_class:
    """
    Funciones que se usaran para ser optimizadas
    """

    def __init__(self, params: dict) -> None:
        name = params["function name"]
        if name == "wood":
            self.f = self.f_wood
            self.gradient = self.gradient_wood
        if name == "rosembrock":
            self.f = self.f_rosembrock
            self.gradient = self.gradient_rosembrock

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
        n = len(x)
        g = zeros(n)
        g[0] = 400*(x[0]*x[0]-x[1])*x[0] + 2*(x[0]-1)
        g[1] = -200*(x[0]*x[0]-x[1])+20.2*(x[1]-1)+19.8*(x[3]-1)
        g[2] = 2*(x[2]-1)+360*(x[2]*x[2]-x[3])*x[2]
        g[3] = -180*(x[2]*x[2]-x[3])+20.2*(x[3]-1)+19.8*(x[1]-1)
        return g

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
