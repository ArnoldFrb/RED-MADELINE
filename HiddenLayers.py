import numpy as np

class HiddenLayers:

    def __init__(self, entradas, salidas, pesos, umbrales, funcActivacion):
        self.entradas = entradas
        self.salidas = salidas
        self.funcActivacion = funcActivacion
        self.pesos = pesos
        self.umbrales = umbrales

    # METODO PARA OBTENER LA FUNCION SOMA
    def FuncionSoma(self, patronPesentado):
        soma = [] # SALIDA DE LA FUNCION SOMA
        for N in range(len(self.pesos)):
            sumatoria = 0     #SUMATORIA DE LA FUNCION SOMA
            for M in range(len(self.pesos[0])):
                sumatoria += (patronPesentado[M] * self.pesos[N][M])
            soma.append(sumatoria - self.umbrales[N])
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

if __name__ == '__main__':
    print('OK')