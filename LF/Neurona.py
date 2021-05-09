import numpy as np


class Neurona:

    def __init__(self, entradas,salidas):
        self.entradas = [[0,0],[0,1],[1,0],[1,1]]
        self.salidas = [[0,0],[0,1],[1,0],[1,1]]

    def FuncionSoma(self, patron, pesos, umbrales):
        soma = []
        for entrada, umbral in zip(patron, umbrales):
            sumatoria = 0
            for peso in pesos:
                sumatoria += entrada + peso
            soma.append(sumatoria - umbral)

        return soma

    def FuncionEscalon(self, salida_soma):
        salida_resultante = []
        for salida in salida_soma:
            salida_resultante.append(1 if salida >= 0 else 0)
        return salida_resultante

    def FuncionLineal(self, salida_soma):
        salida_resultante = salida_soma
        return salida_resultante

    def FuncionSigmoide(self, salida_soma):
        salida_resultante = []
        for salida in salida_soma:
            salida_resultante.append(1 / (1 + np.exp(-salida)))
        return salida_resultante

    def FuncionTangenteH(self, salida_soma):
        salida_resultante = []
        for salida in salida_soma:
            salida_resultante.append(
                (np.exp(salida) - np.exp(-salida)) /
                (np.exp(salida) + np.exp(-salida))
            )
        return salida_resultante
    
    
