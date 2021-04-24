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