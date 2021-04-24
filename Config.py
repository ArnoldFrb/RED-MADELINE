import random as rn
import numpy as np
import pandas as pd
import os
from Layers import *

class Config:

    #CONSTRUCTOR
    def __init__(self):
        self.Entradas = []
        self.Salidas = []
        self.Entranamiento = ''
        self.capas = []
    
    #METODO PARA GENERAR PESOS
    def Generar_pesos(self, row, col):
        Matriz = []
        for N in range(row):
            Fila = []
            for M in range(col):
                Fila.append(round(rn.uniform(-1, 1), 2))
            Matriz.append(Fila)
        return Matriz

    #METODO PARA GENERAR UMBRALES
    def Generar_Umbrales(self, row):
        Fila = []
        for N in range(row):
            Fila.append(round(rn.uniform(-1, 1), 2))
        return Fila

    # LLENAR MATRICES ENTRADAS Y SALIDAS
    def NormalizarDatos(self, ruta):
        Matriz = pd.read_csv(ruta, delimiter=' ')
        col = Matriz.columns
        column = Matriz.to_numpy()
        self.Entranamiento = os.path.basename(os.path.splitext(ruta)[0])

        for i in range(len(col)):
            if 'X' in col[i]:
                Fila = []
                for j in range(len(column)):
                    Fila.append(column[j,i])
                self.Entradas.append(Fila)
            else:
                Fila = []
                for j in range(len(column)):
                    Fila.append(column[j,i])
                self.Salidas.append(Fila)

    # AGREGAR CAPAS OCULTAS
    def AgregarCapas(self, capa, neuronas, funcActivacion):
        encabezado = ['Capa', 'Neuronas', 'Func Activacion']
        self.capas.append([capa, neuronas, funcActivacion])
        
        return pd.DataFrame(data=self.capas, columns=encabezado)

    # INICIAR ENTRENAMIENTO
    def Entrenar(self, rataAprendizaje, errorLineal, numeroIteraciones):

        layers = Layers()
        self.Entradas = np.array(self.Entradas)
        self.Salidas = np.array(self.Salidas)

        s = True
        e = False
        for I in range(len(self.capas)):
            if(I == 0):
                pesos = self.Generar_pesos(len(self.Entradas), self.capas[I][1])
                umblrales = self.Generar_Umbrales(self.capas[I][1])

                for J in range(len(self.Entradas[0])):
                    entrada = self.Entradas[:,J]
                    #salida = np.array([self.Salidas[J]]) if self.Salidas.ndim==1 else (self.Salidas[J,:])
                    func = layers._FuncionActivacion(self.capas[I][2])
                    print(layers.FuncionActivacion(func, layers.FuncionSoma(entrada, pesos, umblrales)))
                    

            if(I > 0 & I < (len(self.capas) - 2)):
                print(I-1, I)
            if(I >= (len(self.capas) - 1)):
                print(I, e)

    # LIMPIAR CAPAS
    def Limpiar(self):
        self.capas = []

if __name__ == '__main__':
    print("Hola") 