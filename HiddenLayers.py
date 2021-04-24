import numpy as np

class HiddenLayers:

    def __init__(self, entradas, salidas, pesos, umbrales, funcActivacion, rataAprendizaje, errorMaximoPermitido, numeroIteraciones):
        self.entradas = entradas
        self.salidas = salidas
        self.pesos = pesos
        self.umbrales = umbrales
        self.funcActivacion = funcActivacion
        self.rataAprendizaje = rataAprendizaje
        self.numeroIteraciones = numeroIteraciones
        self.errorMaximoPermitido = errorMaximoPermitido

    def Entranamiento(self):

        # INICIAR ENTRENAMIENTO
        iteracion = 0
        while True:

            _errorPatron = []

            #CICLO ENCARGADO DE PRESENTAR LOS PATRONES
            for I in range(len(self.entradas)):

                entrada = self.entradas[I,:]
                salida = np.array([self.salidas[I]]) if self.salidas.ndim==1 else (self.salidas[I,:])
                
                errorPatron = np.sum(self.ErrorLineal(salida, self.FuncionSoma(entrada))) / salida.ndim
                _errorPatron.append(errorPatron)

                errorMaximo = np.sum(_errorPatron) / len(self.entradas)

            #CONDICIONES DE PARADA
            if((iteracion > self.numeroIteraciones-1) or (errorMaximo <= self.errorMaximoPermitido)):
                break

    # METODO PARA OBTENER LA FUNCION SOMA
    def FuncionSoma(self, entrada):
        soma = []       # SALIDA DE LA FUNCION SOMA
        for N in range(len(self.pesos)):
            sumatoria = 0       # SUMATORIA DE LA FUNCION SOMA
            for M in range(self.pesos.ndim):
                sumatoria += (entrada[M], self.pesos[M][N])
            soma.append(sumatoria - self.umbrales[N])
        return soma

    def ErrorLineal(self, salida, salidaSoma):
        EL = []          #ERROR LINEAL
        for N in range(len(salidaSoma)):
            EL.append(salida[N] - salidaSoma[N])
        return EL

    #METODO PARA ACTUALIZAR PESOS
    def ActualizarPesos(self, errorLineal, entrada):
        for N in range(len(self.pesos)):
            for M in range(self.pesos.ndim):
                self.pesos[N][M] += (self.rataAprendizaje * errorLineal[N] * entrada[M])

    #METODO PARA ACTUALIZAR UMBRALES
    def ActualizarUmbrales(self, errorLineal):
        for N in range(len(self.umbrales)):
            self.umbrales[N] += (self.rataAprendizaje * errorLineal[N] * 1)

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
    entrenar = HiddenLayers(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), np.array([0.8, -0.6]), np.array([0.1]), 1, 1, 0.1, 2)
    entrenar.Entranamiento()