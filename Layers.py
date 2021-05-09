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

    def ErrorLineal(self, salida, _salida):
        ErrorLienal = []
        for i in range(len(salida)):
            ErrorLienal.append(salida[i] - _salida[i])
        return ErrorLienal

    def ErrorPatron(self, salida, numerodesalidas):
        ErrorPatron = 0
        for i in range(len(salida)):
            ErrorPatron += salida[i]
        ErrorPatron /= numerodesalidas
        return ErrorPatron

    def ActualizarPesosCapas(self, pesos, rataAprendizaje, errorPatron, entradas):
        for i in range(len(pesos)):
            for j in range(len(pesos[0])):
                pesos[i][j] += (rataAprendizaje * errorPatron * entradas[j])
        return pesos

    def ActualizarPesosSalidas(self, pesos, posicion, rataAprendizaje, errorLineal, entradas):
        for j in range(len(pesos)):
            for i in range(len(pesos[0])):
                pesos[j][posicion] += (rataAprendizaje * errorLineal[posicion] * entradas[i])
        return pesos

    def ActualizarUmbralesCapas(self, umbrales, rataAprendizaje, errorPatron):
        for i in range(len(umbrales)):
            umbrales[i] += (rataAprendizaje * errorPatron * 1)
        return umbrales

    def ActualizarUmbralesSalidas(self, umbrales, posicion, rataAprendizaje, errorLineal):
        for i in range(len(umbrales)):
                umbrales[posicion] += (rataAprendizaje * errorLineal[posicion] * 1)
        return umbrales

    def _FuncionActivacion(self, e):
        op = 0
        if ("SIGMOIDE" == e):
            op = 1

        if ("ESCALON" == e):
            op = 2

        if ("LINEAL" == e):
            op = 3

        if ("TANGENTE H." == e):
            op = 2

        if ("GAUSSIANA" == e):
            op = 3

        return op

    #NOMBRE DE LA FUNCION SALIDA
    def FuncionActivacionCapas(self, func, salidaSoma):
        switcher = {
            1: self.FuncionSigmoide(salidaSoma),
            2: self.FuncionTangenteH(salidaSoma),
            3: 'SIGMOIDE'
        }
        return switcher.get(func, "ERROR")

    def FuncionActivacionSalidas(self, func, salidaSoma):
        switcher = {
            1: self.FuncionSigmoide(salidaSoma),
            2: self.FuncionEscalon(salidaSoma),
            3: self.FuncionLineal(salidaSoma)
        }
        return switcher.get(func, "ERROR")

    #METODO PARA OBTENER LA FUNCION ESCALON
    def FuncionEscalon(self, salidaSoma):
        Yr = []
        for N in range(len(salidaSoma)):
            Yr.append(1 if salidaSoma[N]>=0 else 0)
        return Yr

    #METODO PARA OBTENER LA FUNCION LINEAL
    def FuncionLineal(self, salidaSoma):
        Yr = salidaSoma
        return Yr

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