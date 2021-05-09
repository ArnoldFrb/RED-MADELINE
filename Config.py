import random as rn
import numpy as np
import pandas as pd
import os
from Layers import *
from Views import *
from heapq import nsmallest
import tkinter as tk


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
    def Entrenar(self, rataAprendizaje, errorMaximo, numeroIteraciones, funcionSalida):

        layers = Layers()
        self.Entradas = self.NormalizarEntradas(self.Entradas)
        self.Salidas = self.NormalizarSalidas(self.Salidas) if len(self.Salidas)==1 else self._NormalizarSalidas(self.Salidas)

        DimensionPesos = []
        DimensionUmbrales = []
        for I in range(len(self.capas)):

            # CONDICION PARA ENTRADA Y CAPA 1
            if(I == 0):
                DimensionPesos.append(self.Generar_pesos(len(self.Entradas[0]), self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
             # CONDICION PARA CAPAS INTERMEDIAS
            if(I > 0 & I < (len(self.capas) - 1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I-1][1], self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
            # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
            if(I >= (len(self.capas)-1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I][1], len(self.Salidas[0])))
                DimensionUmbrales.append(self.Generar_Umbrales(len(self.Salidas[0])))
        
        for x in range(len(self.Entradas)): # ARRAY PARA PRESENTAR PATRONES

            entrada = self.Entradas[x][:]
            salida = self.Salidas[x][:]

            print('//////////////////')
            print('////////////////////////////////////')
            print('PATRON', x+1)
            print()
            
            NuevaEntrada = []

            for y in range(len(self.capas)): # ARRAY PARA RECORER CAPAS

                if(y == 0):
                    print('ENTRADAS x CAPA', y,'=>' , len(entrada), 'x', self.capas[y][1])

                    print('ENTRADAS')
                    print(np.array(entrada))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[y][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[y][:][:]))
                    print()

                    print('FUNCION ACTIVACION NO LINEAL:', self.capas[y][2])
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(self.capas[y][2]), 
                            layers.FuncionSoma(entrada, DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
                    print(np.array(NuevaEntrada))
                    print()
                    print('----------------------------------------')
                    print()

                if(y > 0 & y < (len(self.capas) - 1)):
                    print('CAPA', y-1, 'x CAPA', y,'=>' , self.capas[y-1][1], 'x', self.capas[y][1])
                    
                    print('ENTRADAS')
                    print(np.array(NuevaEntrada[y-1]))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[y][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[y][:][:]))
                    print()

                    print('FUNCION ACTIVACION NO LINEAL:', self.capas[y][2])
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(self.capas[y][2]), 
                            layers.FuncionSoma(NuevaEntrada[y-1], DimensionPesos[y][:][:], DimensionUmbrales[y][:][:])))
                    print(np.array(NuevaEntrada[y]))
                    print()
                    print('----------------------------------------')
                    print()

                if(y >= (len(self.capas)-1)):
                    print('CAPA', y, 'x SALIDAS','=>' , self.capas[y][1], 'x', len(salida))
                    
                    print('ENTRADAS')
                    print(np.array(NuevaEntrada[y]))
                    print()

                    print('PESOS')
                    print(np.array(DimensionPesos[len(self.capas)][:][:]))
                    print()

                    print('UMBRALES')
                    print(np.array(DimensionUmbrales[len(self.capas)][:][:]))
                    print()

                    print('FUNCION ACTIVACION LIENAL:', funcionSalida)
                    NuevaEntrada.append(
                        layers.FuncionActivacionCapas(
                            layers._FuncionActivacion(funcionSalida), 
                            layers.FuncionSoma(NuevaEntrada[y], DimensionPesos[len(self.capas)][:][:], DimensionUmbrales[len(self.capas)][:][:])))
                    print(np.array(NuevaEntrada[y+1]))
                    print()

                    print('ERROR LINEAL')
                    ErrorLineal = layers.ErrorLineal(salida, NuevaEntrada[y+1])
                    print(ErrorLineal)
                    print()

                    print('ERROR PATRON')
                    _ErrorPatron = sum(ErrorLineal) / len(ErrorLineal)
                    print(_ErrorPatron)
                    print()

                    _DimensionPesos = DimensionPesos
                    _DimensionUmbrales = DimensionUmbrales
                    _NuevaEntradaTemporal = NuevaEntrada
                    
                    CapasInversa = self.capas[::-1]
                    DimensionPesos = DimensionPesos[::-1]
                    DimensionUmbrales = DimensionUmbrales[::-1]
                    NuevaEntrada = NuevaEntrada[::-1]

                    _PesosTempo = 0
                    _UmbralesTempo = 0

                    print('///////////////////////////////////////')
                    print()

                    for z in range(len(self.capas)):

                        if(z == 0):
                            ErrorLienalMenor = np.amin(ErrorLineal)
                            indexM = ErrorLineal.index(ErrorLienalMenor)
                            print('ERROR LINEAL MENOR:', ErrorLienalMenor, 'INDEX:', indexM)
                            print()

                            print('PESOS TEMPORALES')
                            PesosTemporales = layers.ActualizarPesosSalidas(
                                DimensionPesos[z][:][:], indexM, int(rataAprendizaje), ErrorLineal, entrada)
                            print(np.array(PesosTemporales))
                            print()

                            print('UMBRALES TEMPORALES')
                            UmbralesTemporales = layers.ActualizarUmbralesSalidas(
                                DimensionUmbrales[z][:][:], indexM, int(rataAprendizaje), ErrorLineal)
                            print(np.array(UmbralesTemporales))
                            print()

                            print('NUEVO FUNCION ACTIVACION LIENAL:', funcionSalida)
                            _NuevaEntrada = layers.FuncionActivacionCapas(
                                layers._FuncionActivacion(funcionSalida), 
                                layers.FuncionSoma(NuevaEntrada[y], PesosTemporales, UmbralesTemporales))
                            print(np.array(_NuevaEntrada))
                            print()

                            print('NUEVO ERROR LINEAL')
                            _ErrorLineal = layers.ErrorLineal(salida, _NuevaEntrada)
                            print(_ErrorLineal)
                            print()

                            print('ES MENOR EL NUEVO ERROR:', _ErrorLineal[indexM] < ErrorLineal[indexM])
                            print()

                            if(_ErrorLineal[indexM] < ErrorLineal[indexM]):
                                DimensionPesos[z][:][:] = PesosTemporales
                                DimensionUmbrales[z][:][:] = UmbralesTemporales
                                print('PESOS ACTUALIZADOS')
                                print(np.array(DimensionPesos[z][:][:]))
                                print()

                                print('UMBRALES ACTUALIZADOS')
                                print(np.array(DimensionUmbrales[z][:][:]))
                                print()
                            else:
                                print('NO SE ACTUALIZAN PESOS Y UMBRALES')
                                print('PESOS')
                                print(np.array(DimensionPesos[z][:][:]))
                                print()

                                print('UMBRALES')
                                print(np.array(DimensionUmbrales[z][:][:]))
                                print()

                            print('///////////////////////////////////////////////////////////////////')
                            print('///////////////////////////////////////////////////////////////////')
                            print()

                        if(z > 0 & z < (len(self.capas) - 1)):

                            print(np.array(ErrorLineal))
                            print(np.array(_DimensionPesos[z][:][:]))
                            
                            ErrorNoLienal = layers.ErrorNoLineal(ErrorLineal, DimensionPesos[z+1][:][:])
                            ErrorNoLienal_ = ErrorNoLienal
                            index = len(ErrorNoLienal)

                            print(np.array(DimensionPesos[z][:][:]))
                            print()
                            print(np.array(DimensionUmbrales[z][:][:]))
                            print()

                            _PesosTempo = _DimensionPesos[z][:][:]
                            _UmbralesTempo = _DimensionUmbrales[z][:][:]

                            print('///////////////////////////////////////////////////////////////////')
                            print('///////////////////////////////////////////////////////////////////')
                            print()

                            for i in range(index):
                                _ErrorNoLienal = nsmallest(i+1, ErrorNoLienal, key=lambda x: abs(x-0))
                                if _ErrorNoLienal:
                                    for cero in _ErrorNoLienal:
                                        print('ERROR NO LINEAL')
                                        print(ErrorNoLienal_)
                                        print()

                                        print('ERRORES NO LINEALES', i+1)
                                        print(_ErrorNoLienal)
                                        print()

                                        indexC = ErrorNoLienal_.index(cero)
                                        print('ERROR NO LINEAL CERCANO A CERO:', cero, 'INDEX:', indexC)
                                        print()

                                        print('///////////////////////////////////////////////////////////////////')
                                        print('///////////////////////////////////////////////////////////////////')
                                        print('BUSCANDO EL ERROR')
                                        print()

                                        print(np.array(_PesosTempo))
                                        print()
                                        print(np.array(_UmbralesTempo))
                                        print()

                                        print('///////////////////////////////////////////////////////////////////')
                                        print('///////////////////////////////////////////////////////////////////')
                                        print()

                                        PesosTs = layers.ActualizarPesosCapas(
                                            _DimensionPesos[z][:][:], indexC, int(rataAprendizaje), _ErrorPatron, _NuevaEntradaTemporal[z])
                                            
                                        UmbralesTs = layers.ActualizarUmbralesCapas(
                                            _DimensionUmbrales[z][:][:], indexC, int(rataAprendizaje), _ErrorPatron)

                                        print('///////////////////////////////////////////////////////////////////')
                                        print('///////////////////////////////////////////////////////////////////')
                                        print('BUSCANDO EL ERROR')
                                        print()

                                        print(np.array(_DimensionPesos[z][:][:]))
                                        print()
                                        print(np.array(_DimensionUmbrales[z][:][:]))
                                        print()

                                        print('///////////////////////////////////////////////////////////////////')
                                        print('///////////////////////////////////////////////////////////////////')
                                        print()

                                        ind = 0
                                        for j in range(len(self.capas)):
                                            if (DimensionPesos[z] == _DimensionPesos[j]):
                                                ind = j
                                        
                                        for j in range(ind, len(self.capas)):
                                            if(j > 0 & j < (len(self.capas) - 1)):

                                                print('--ACTUALIZAR PESOS Y UMBRALES--')
                                                print()

                                                print('CAPA', j-1, 'x CAPA', j,'=>' , self.capas[j-1][1], 'x', self.capas[j][1])
                                                
                                                print('ENTRADAS')
                                                print(np.array(_NuevaEntradaTemporal[j-1]))
                                                print()

                                                print('PESOS TEMPORALES CAPAS')
                                                print(np.array(PesosTs))
                                                print()

                                                print('UMBRALES TEMPORALES CAPAS')
                                                print(np.array(UmbralesTs))
                                                print()

                                                print('FUNCION ACTIVACION NO LINEAL:', self.capas[j][2])
                                                _NuevaEntradaTemporal[j] = (
                                                    layers.FuncionActivacionCapas(
                                                        layers._FuncionActivacion(self.capas[j][2]), 
                                                        layers.FuncionSoma(_NuevaEntradaTemporal[j-1], PesosTs, UmbralesTs)))
                                                print(np.array(_NuevaEntradaTemporal[j]))
                                                print()

                                            if(j >= (len(self.capas)-1)):
                                                print('CAPA', y, 'x SALIDAS','=>' , self.capas[y][1], 'x', len(salida))
                                                
                                                print('ENTRADAS')
                                                print(np.array(_NuevaEntradaTemporal[j]))
                                                print()

                                                print('PESOS')
                                                print(np.array(_DimensionPesos[len(self.capas)][:][:]))
                                                print()

                                                print('UMBRALES')
                                                print(np.array(_DimensionUmbrales[len(self.capas)][:][:]))
                                                print()

                                                print('FUNCION ACTIVACION LIENAL:', funcionSalida)
                                                _NuevaEntradaTemporal[j+1] = (
                                                    layers.FuncionActivacionCapas(
                                                        layers._FuncionActivacion(funcionSalida), 
                                                        layers.FuncionSoma(_NuevaEntradaTemporal[j], _DimensionPesos[len(self.capas)][:][:], _DimensionUmbrales[len(self.capas)][:][:])))
                                                print(np.array(_NuevaEntradaTemporal[j+1]))
                                                print()

                                                print('ERROR LINEAL')
                                                _ErrorLineal = layers.ErrorLineal(salida, _NuevaEntradaTemporal[j+1])
                                                print(_ErrorLineal)
                                                print()

                                                NuevoErrorNoLienal = layers.ErrorNoLineal(_ErrorLineal, DimensionPesos[z-1][:][:])

                                                print(indexC, 'NUEVO ERROR:', NuevoErrorNoLienal[indexC], '<', 'ANTIGUO ERROR:', ErrorNoLienal_[indexC])
                                                print()

                                                if (NuevoErrorNoLienal[indexC] < ErrorNoLienal_[indexC]):
                                                    DimensionPesos[z][:][:] = PesosTs
                                                    DimensionUmbrales[z][:][:] = UmbralesTs
                                                    print('PESOS ACTUALIZADOS')
                                                    print(np.array(DimensionPesos[z][:][:]))
                                                    print()

                                                    print('UMBRALES ACTUALIZADOS')
                                                    print(np.array(DimensionUmbrales[z][:][:]))
                                                    print()
                                                else:
                                                    print('LOS PESOS Y UMBRALES NO SON ACTAULIZADOS')
                                                    print()
                                                    print('PESOS')
                                                    print(np.array(_PesosTempo))
                                                    print()

                                                    print('UMBRALES')
                                                    print(np.array(_UmbralesTempo))
                                                    print()
                                        
                                        print('///////////////////////////////////////////////////////////////////')
                                        print()

                                        ErrorNoLienal = list( filter(lambda x: x != cero, ErrorNoLienal) )

                            print('///////////////////////////////////////////////////////////////////')
                            print('///////////////////////////////////////////////////////////////////')
                            print()

                        if(z >= (len(self.capas)-1)):
                            print('Y')

                    DimensionPesos = DimensionPesos[::-1]
                    DimensionUmbrales = DimensionUmbrales[::-1]
                    NuevaEntrada = NuevaEntrada[::-1]

        print()
        print('//////////// - FIN ENTRENAMIENTO - ////////////')
        print()

    # LIMPIAR CAPAS
    def Limpiar(self):
        self.capas = []

    def NormalizarEntradas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            aux = []
            for j in range(len(salida)):
                aux.append(salida[j][i])
            salidas.append(aux)
        return salidas

    def NormalizarSalidas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            s = []
            for j in range(len(salida)):
                s.append(salida[j][i])
            salidas.append(s)
        return salidas

    def _NormalizarSalidas(self, salida):
        salidas = []
        for i in range(len(salida[0])):
            aux = []
            for j in range(len(salida)):
                aux.append(salida[j][i])
            salidas.append(aux)
        return salidas

if __name__ == '__main__':
    winw = tk.Tk()
    Views(winw)
    winw.mainloop()