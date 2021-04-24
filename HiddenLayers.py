import numpy as np
import random as rn

class HiddenLayers:

    def __init__(self, rataAprendizaje, errorMaximoPermitido, numeroIteraciones):
        self.entradas = []
        self.salidas = []
        self.pesos = []
        self.umbrales = []
        self.funcActivacion = 0
        self.rataAprendizaje = rataAprendizaje
        self.numeroIteraciones = numeroIteraciones
        self.errorMaximoPermitido = errorMaximoPermitido

    def Entranamiento(self, entradas, salidas, pesos, umbrales, funcActivacion):

        self.entradas = entradas
        self.salidas = salidas
        self.pesos = pesos
        self.umbrales = umbrales

        # INICIAR ENTRENAMIENTO
        iteracion = 0
        while True:

            _errorPatron = []

            #CICLO ENCARGADO DE PRESENTAR LOS PATRONES
            for I in range(len(self.entradas)):

                entrada = self.entradas[I,:]
                salida = np.array([self.salidas[I]]) if self.salidas.ndim==1 else (self.salidas[I,:])

                res = self.FuncionSoma(entrada)
                
            #CONDICIONES DE PARADA
            if(iteracion > self.numeroIteraciones-1):
                break

        return res
        

    # METODO PARA OBTENER LA FUNCION SOMA
    def FuncionSoma(self, entrada):
        soma = []       # SALIDA DE LA FUNCION SOMA
        for N in range(len(self.pesos)):
            sumatoria = 0       # SUMATORIA DE LA FUNCION SOMA

            for M in range(len(self.pesos[0])):
                sumatoria += (entrada[M] + self.pesos[N][M])

            soma.append(sumatoria - self.umbrales[N])
        return soma

    def ErrorLineal(self, salida, salidaSoma):
        EL = []     # ERROR LINEAL
        for N in range(len(salidaSoma)):
            print(salida, salidaSoma)
        return EL

    def _FuncionActivacion(self, e):
        if ("SIGMOIDE" == e):
            return 1

        if ("TANGENTE H." == e):
            return 2

        if ("GAUSSIANA" == e):
            return 3

    #NOMBRE DE LA FUNCION SALIDA
    def FuncionActivacion(self, func, salidaSoma):
        switcher = {
            1: self.FuncionSigmoide(salidaSoma),
            2: self.FuncionTangenteH(salidaSoma),
            3: 'SIGMOIDE'
        }
        return switcher.get(func, "ERROR")

    #METODO PARA OBTENER LA FUNCION SIGMOIDE
    def FuncionSigmoide(self, salidaSoma):
        Yr = []
        for N in range(len(salidaSoma)):
            Yr.append(1 / (1 + np.exp(-salidaSoma[N])))
        return Yr

    #METODO PARA OBTENER LA FUNCION TANGENTE H.
    def FuncionTangenteH(self, salidaSoma):
        Yr = []
        for N in range(len(salidaSoma)):
            Yr.append((np.exp(salidaSoma[N]) - np.exp(-salidaSoma[N])) / (np.exp(salidaSoma[N]) + np.exp(-salidaSoma[N])))
        return Yr

if __name__ == '__main__':
        print('Ok')


entrenar = HiddenLayers(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), np.array([0.8, -0.6]), np.array([0.1]), 1, 1, 0.1, 2)
entrenar.Entranamiento()

# [1][1][1][1][1]_[1][1][1]
# [1][0][1][1][0]_[1][1][1]
# [1][1][0][1][1]_[1][1][1]


p = 3
x = [[1,1,1,1,1],[1,0,1,1,0],[1,1,0,1,1]]
y = [[1,1,1],[1,1,1],[1,1,1]]
n1 = 7
n2 = 5
#
w1 = [[-1,0,1,1,0,1,1],[-1,0,1,1,0,1,0],[-1,0,1,1,0,1,0],[-1,0,1,1,0,1,0],[-1,0,1,1,0,1,0]]
w2 = [[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0]]
w3 = [[-1,0,1,1,0],[-1,0,1,1,0],[-1,0,1,1,0]]
#
u = [[1],[1],[1],[1],[1],[1],[1]]
u = [[1],[1],[1],[1],[1]]
u = [[1],[1],[1]]
#
r = 1
e = 0.1
#
# class HiddenLayers:
#     def __init__(self, entradas, salidas, pesos, umbrales, funcActivacion, rataAprendizaje, errorMaximoPermitido, numeroIteraciones):
#         self.entradas = entradas
#         self.salidas = salidas
#         self.pesos = pesos
#         self.umbrales = umbrales
#         self.funcActivacion = funcActivacion
#         self.rataAprendizaje = rataAprendizaje
#         self.numeroIteraciones = numeroIteraciones
#         self.errorMaximoPermitido = errorMaximoPermitido
#     def SalidaNuevaCapa (self, entradas, n_capa, pesos, funcion):
#         sum = 0
#         for j in range(len(entradas[0])) :
#             for i in range(n_capa) :
#                 sum += (entradas[j] * pesos[j][i]) - umbrales[i]
#         return funcion(sum)
#     def 