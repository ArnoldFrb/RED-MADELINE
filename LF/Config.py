import random as rn
# import numpy as np
# import pandas as pd
# import os
from Neurona import *
class Config:

    #CONSTRUCTOR
    def __init__(self,entradas, salidas):
        self.neurona = Neurona(entradas, salidas)
        self.red = []


    def GenerarPesos(self, numero_neurona):
        matriz = []
        for i in self.salidas:
            fila = []
            for j in range(numero_neurona):
                fila.append(round(rn.uniform(-1, 1), 2))
            matriz.append(fila)
        return matriz


    def GenerarUmbrales(self, numero_neurona):
        Fila = []
        for N in range(numero_neurona):
            Fila.append(round(rn.uniform(-1, 1), 2))
        return Fila 

