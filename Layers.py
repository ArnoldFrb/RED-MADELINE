import numpy as np
import random as rn

class Layers:

    def __init__(self):
        print()

    def FuncionSoma(self, entrada, pesos, umbrales):
        soma = []
        for i in range(len(pesos[0])):
            sumatoria = 0
            for j in range(len(pesos)):
                sumatoria = (entrada[j] + pesos[j][i])
            soma.append(sumatoria - umbrales[i])
        return soma

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