# import random as rn

# def InicializarCapas(numeroCapas, rangoA, rangoB):
#     capas = []
#     for I in range(numeroCapas):
#         filas = []
#         filas.append(I+1)
#         filas.append(round(rn.uniform(rangoA, rangoB)))
#         capas.append(filas)

#     print(capas)

# InicializarCapas(10, 10, 15)

print(0.018 < 0.06)

EntradasCapas = []
        _EntradasCapas = []
        _ErrorPatron = []

        DimensionPesos = []
        DimensionUmbrales = []
        DimensionSalidas = []

        for I in range(len(self.capas)):

            # CONDICION PARA ENTRADA Y CAPA 1
            if(I == 0):
                DimensionPesos.append(self.Generar_pesos(len(self.Entradas), self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
             # CONDICION PARA CAPAS INTERMEDIAS
            if(I > 0 & I < (len(self.capas) - 1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I-1][1], self.capas[I][1]))
                DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
            # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
            if(I >= (len(self.capas) - 1)):
                DimensionPesos.append(self.Generar_pesos(self.capas[I][1], len(self.Salidas[0])))
                DimensionUmbrales.append(self.Generar_Umbrales(len(self.Salidas[0])))

        for I in range(len(self.capas)):

            # CONDICION PARA ENTRADA Y CAPA 1
            if(I == 0):
                print('ENTRADAS x CAPA:', I, '=', len(self.Entradas), 'x', self.capas[I][1])
                print()

                for J in range(len(self.Entradas[0])):

                    entrada = self.Entradas[:,J]

                    func = layers._FuncionActivacion(self.capas[I][2])
                    EntradasCapas.append(
                        layers.FuncionActivacionCapas(func, 
                        layers.FuncionSoma(entrada, DimensionPesos[I][:][:], DimensionUmbrales[I][:][:])))

                DimensionSalidas.append(EntradasCapas)

                print(np.array(EntradasCapas))
                print()
                print()
                
             # CONDICION PARA CAPAS INTERMEDIAS
            if(I > 0 & I < (len(self.capas) - 1)):
                print('CAPA', I-1, 'x CAPA:', I, '=', self.capas[I-1][1], 'x', self.capas[I][1])
                print()

                for J in range(len(EntradasCapas)):

                    if( J == 0 ):
                        _EntradasCapas = EntradasCapas
                        EntradasCapas = []

                    entradasCapas = _EntradasCapas[J][:]

                    func = layers._FuncionActivacion(self.capas[I][2])
                    EntradasCapas.append(
                        layers.FuncionActivacionCapas(func, 
                        layers.FuncionSoma(entradasCapas, DimensionPesos[I][:][:], DimensionUmbrales[I][:][:])))

                DimensionSalidas.append(EntradasCapas)

                print(np.array(EntradasCapas))
                print()
                print()
                
            # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
            if(I >= (len(self.capas) - 1)):
                print('CAPA', I, 'x ENTRADAS', '=', self.capas[I][1], 'x', len(self.Salidas[0]))
                print()

                for J in range(len(EntradasCapas)):

                    if( J == 0 ):
                        _EntradasCapas = EntradasCapas
                        EntradasCapas = []

                    entradasCapas = _EntradasCapas[J][:]

                    func = layers._FuncionActivacion(funcionSalida)
                    
                    if (len(self.Salidas)==1):
                        salida = self.Salidas[J][:]
                    else:
                        salida = self.Salidas[:][J]

                    

                    ErrorLineal = layers.ErrorLineal(salida, 
                        layers.FuncionActivacionSalidas(func, 
                        layers.FuncionSoma(entradasCapas, DimensionPesos[I][:][:], DimensionUmbrales[I][:][:])))

                    print('EL', np.array(ErrorLineal))

                    ErrorPatron = layers.ErrorPatron(ErrorLineal, len(self.Salidas[0]))

                DimensionSalidas.append(EntradasCapas)

        # for I in range(len(self.capas)):

        #     # CONDICION PARA ENTRADA Y CAPA 1
        #     if(I == 0):
        #         DimensionPesos.append(self.Generar_pesos(len(self.Entradas), self.capas[I][1]))
        #         DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
        #      # CONDICION PARA CAPAS INTERMEDIAS
        #     if(I > 0 & I < (len(self.capas) - 1)):
        #         DimensionPesos.append(self.Generar_pesos(self.capas[I-1][1], self.capas[I][1]))
        #         DimensionUmbrales.append(self.Generar_Umbrales(self.capas[I][1]))
                
        #     # CONDICION PARA LA ULTIMA CAPA Y SALIDAS
        #     if(I >= (len(self.capas) - 1)):
        #         DimensionPesos.append(self.Generar_pesos(self.capas[I][1], len(self.Salidas[0])))
        #         DimensionUmbrales.append(self.Generar_Umbrales(len(self.Salidas[0])))