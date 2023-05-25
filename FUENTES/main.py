import numpy as np
import time
import csv

# Semilla para la generación de números aleatorios
rng = np.random.default_rng(0)

# Función auxiliar para convertir los string que sean números a float
def toFloat(x):
    try:
        return float(x)
    except ValueError:
        return x
    

# Función auxiliar para dar formato a los números
def formatNumber(x):
    # Redondeamos a dos decimales y cambiamos el punto por la coma
    return "{:.2f}".format(x).replace('.', ',')
    

# Función para calcular el algoritmo kNN (con k = 1)
# param X = conjunto de datos
# param Y = elementos a clasificar
# param W = pesos
# return pos = vector con las posiciones de los elementos más cercanos de X a cada elemento de Y
def oneNN(X,Y,W):
    X = np.array(X)
    Y = np.array(Y)
    W = np.array(W)

    # No se consideran las características con pesos menores que 0.1 (reducción)
    W[W < 0.1] = 0

    # Calculamos una matriz, donde las columnas son las distancias de cada elemento de Y a cada elemento de X
    distancias = np.sqrt(np.sum(np.multiply(W, np.square(X[:, np.newaxis] - Y)), axis=2))
    # Ignoramos la distancia a sí mismo
    distancias = np.where(distancias==0, np.max(distancias)+1, distancias)

    # Obtenemos un vector con las posiciones de los elementos más cercanos de X a cada elemento de Y
    pos = np.argmin(distancias, axis=0)

    return pos


# Función para leer ficheros ARFF y obtener los datos
def readARFF(filename):
    f = open(filename, 'r')
    data = []

    for line in f:
        # Comprueba que la línea no sea vacía
        if line.strip():
            # Comprueba que la línea no sea un comentario
            if line[0] != '@' and line[0] != '%':
                # Hacemos un vector con los distintos atributos de la línea y lo añadimos a data
                data.append([toFloat(l.rstrip()) for l in line.split(',')])

    f.close()

    return data


# Función para normalizar los datos de un array dado
# param X = matriz de datos
def normalize(X):
    X = np.array(X)
    
    # Calculamos un vector con los valores mínimos por columna
    min = np.min(X, axis=0)
    # Calculamos un vector con los valores máximos por columna
    max = np.max(X, axis=0)
    # Normalizamos los datos
    X = (X - min) / (max - min)

    return X


# Implementación de la tasa de clasificación
# param Xtrain = datos entrenamiento
# param Xtest = datos test
# param Ytrain = clases entrenamiento
# param Ytest = clases test
# param W = pesos
# return tasa_clas = tasa de clasificación
def tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W):
    Xtrain = np.array(Xtrain)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    Ytrain = np.array(Ytrain)
    W = np.array(W)

    # Obtenemos las posiciones de los elementos más cercanos de Xtrain a cada elemento de Xtest
    pos = oneNN(Xtrain, Xtest, W)
    # Calculamos el número de aciertos
    aciertos = np.sum(Ytrain[pos] == Ytest)
    tasa_clas = 100.0 * aciertos / Xtest.shape[0]

    return tasa_clas


# Implementación de la tasa de reducción
# param W = pesos
# return tasa_red = tasa de reducción
def tasaReduccion(W):
    W = np.array(W)
    valores_red = 0

    valores_red = np.sum(W < 0.1)
    tasa_red = 100.0 * valores_red / W.shape[0]

    return tasa_red


# Implementacion de la función de fitness que queremos maximizar (en función de W)
# param tasas_clas = tasa de clasificación
# param tasa_red = tasa de reducción
# return fit = fitness
def fitness(tasa_clas, tasa_red):
    # Tomamos alpha = 0.8
    fit = 0.8 * tasa_clas + 0.2 * tasa_red

    return fit


# Implementación del algoritmo de búsqueda local
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# param tope = número máximo de evaluaciones de la función objetivo
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def busquedaLocal(Xtrain, Ytrain, Xtest, Ytest, tope=15000):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    mutadas = np.ones(num_caract)
    eval = 0
    generados = 0

    # Generamos una solución inicial aleatoria
    W = rng.random(num_caract)

    # Calculamos el fitness de la solución inicial
    fit = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, W), tasaReduccion(W))


    # Detenemos BL si no se encuentra mejora tras generar 20*num_caract vecinos o si se han realizado tope evaluaciones de la función objetivo
    while generados < 20 * num_caract and eval < tope:
        # Mutamos una componente aleatoria sin repetición (emulamos un bucle do-while)
        while True:
            pos = rng.integers(num_caract)

            if mutadas[pos] == 1:
                mutadas[pos] = 0
                break

        Wmod = W.copy()
        # Usamos una mutación normal con media 0 y desviación típica 0.3, después truncamos para que el valor esté en [0,1]
        Wmod[pos] += rng.normal(scale=0.3)
        if Wmod[pos] < 0:
            Wmod[pos] = 0.0
        elif Wmod[pos] > 1:
            Wmod[pos] = 1.0

        generados += 1

        # Calculamos el fitness de la solución mutada
        fitmod = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, Wmod), tasaReduccion(Wmod))
        eval += 1

        # Si el fitness de la solución mutada es mejor que el de la solución inicial, la solución inicial pasa a ser la mutada
        if fitmod > fit:
            W = Wmod
            fit = fitmod
            mutadas = np.ones(num_caract)
            generados = 0
        elif generados % num_caract == 0:
            mutadas = np.ones(num_caract)


    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)

    return W, fit, tasa_clas, tasa_red, elapsed



# Método basado en BL ya implementada, pero con entradas y salidas adaptadas para acloparlo mejor a las mh a implementar
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param W = solución inicial
# param tope = número máximo de evaluaciones de la función objetivo
# param K = número de componentes a mutar (vecindario)
# return W = pesos, fit = fitness
def busquedaLocalAux(Xtrain, Ytrain, W, tope, K=1):
    num_caract = Xtrain.shape[1]
    posDisponibles = np.arange(num_caract)
    eval = 0
    generados = 0

    # Calculamos el fitness de la solución inicial
    fit = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, W), tasaReduccion(W))


    # Detenemos BL si no se encuentra mejora tras generar 20*num_caract vecinos o si se han realizado tope evaluaciones de la función objetivo
    while generados < 20 * num_caract and eval < tope:
        # Mutamos K componentes aleatorias sin repetición
        pos = rng.choice(posDisponibles, K, replace=False)
        posDisponibles = np.setdiff1d(posDisponibles, pos)

        Wmod = W.copy()
        # Usamos una mutación normal con media 0 y desviación típica 0.3, después truncamos para que los valores estén en [0,1]
        Wmod[pos] += rng.normal(scale=0.3, size=K)
        Wmod = np.clip(Wmod, 0, 1)

        generados += 1

        # Calculamos el fitness de la solución mutada
        fitmod = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, Wmod), tasaReduccion(Wmod))
        eval += 1

        # Si el fitness de la solución mutada es mejor que el de la solución inicial, la solución inicial pasa a ser la mutada
        if fitmod > fit:
            W = Wmod
            fit = fitmod
            posDisponibles = np.arange(num_caract)
            generados = 0
        # Si en la siguiente iteración no se van a poder tomar K componentes, se resetea el vector de posiciones disponibles
        elif posDisponibles.size < K:
            posDisponibles = np.arange(num_caract)


    return W, fit



# Implementación del algoritmo de enfriamiento simulado
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def ES(Xtrain, Ytrain, Xtest, Ytest):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    phi = 0.2
    mu = 0.3
    tempFinal = 0.0001
    max_vecinos = 10 * num_caract
    vecinos = 0
    max_exitos = 0.1 * max_vecinos
    exitos = 0
    max_eval = 15000
    eval = 0
    M = max_eval / max_vecinos

    # Generamos una solución aleatoria
    W = rng.random(num_caract)
    fit = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, W), tasaReduccion(W))

    # Inicializamos la temperatura
    temp = mu * fit / -np.log(phi)

    # Comprobamos siempre que la temperatura final sea menor que la inicial
    while temp < tempFinal:
        # Si no lo es, modificamos la temperatura final hasta que lo sea ?????????????????????
        tempFinal *= 0.1

    # Bucle principal
    while temp > tempFinal and eval < max_eval:
        vecinos = 0
        exitos = 0

        # Bucle interno
        while vecinos < max_vecinos and exitos < max_exitos:
            # Mutamos una componente aleatoria
            pos = rng.integers(num_caract)
            Wmod = W.copy()

            # Usamos una mutación normal con media 0 y desviación típica 0.3, después truncamos para que los valores estén en [0,1]
            Wmod[pos] += rng.normal(scale=0.3)
            if Wmod[pos] < 0:
                Wmod[pos] = 0.0
            elif Wmod[pos] > 1:
                Wmod[pos] = 1.0

            vecinos += 1

            # Calculamos el fitness de la solución mutada
            fitmod = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, Wmod), tasaReduccion(Wmod))
            eval += 1

            # En las transparencias se quiere minimizar, pero en este caso queremos maximizar, por lo que cambiamos el signo
            dif_fit = fit - fitmod

            # Si la solución mutada es mejor que la solución actual, la solución actual pasa a ser la mutada
            if dif_fit < 0:
                W = Wmod
                fit = fitmod
                exitos += 1
            # Si no lo es, todavía hay una probabilidad de que la solución actual pase a ser la mutada
            elif rng.random() < np.exp(-dif_fit / temp):
                W = Wmod
                fit = fitmod
                # Aquí también se incrementa el número de éxitos????????????????
                exitos += 1


        # Esquema de enfriamiento: esquema de Cauchy modificado
        beta = (temp - tempFinal) / (M * temp * tempFinal)
        temp = temp / (1 + beta * temp)

        # Si el número de éxitos en el enfriamiento actual es 0, paramos ?????????????????????
        if exitos == 0:
            break

    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)

    return W, fit, tasa_clas, tasa_red, elapsed


# Método basado en ES ya implementado, pero con entradas y salidas adaptadas para acloparlo mejor a las mh a implementar
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param W = solución inicial
# param tope = número máximo de evaluaciones
# return W = pesos, fit = fitness
def ESAux(Xtrain, Ytrain, W, tope):
    num_caract = Xtrain.shape[1]
    phi = 0.2
    mu = 0.3
    tempFinal = 0.0001
    max_vecinos = 10 * num_caract
    vecinos = 0
    max_exitos = 0.1 * max_vecinos
    exitos = 0
    max_eval = tope
    eval = 0
    M = max_eval / max_vecinos

    # Calculamos el fitness del W dado
    fit = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, W), tasaReduccion(W))

    # Inicializamos la temperatura
    temp = mu * fit / -np.log(phi)

    # Comprobamos siempre que la temperatura final sea menor que la inicial
    while temp < tempFinal:
        # Si no lo es, modificamos la temperatura final hasta que lo sea
        tempFinal *= 0.1

    # Bucle principal
    while temp > tempFinal and eval < max_eval:
        vecinos = 0
        exitos = 0

        # Bucle interno
        while vecinos < max_vecinos and exitos < max_exitos:
            # Mutamos una componente aleatoria
            pos = rng.integers(num_caract)
            Wmod = W.copy()

            # Usamos una mutación normal con media 0 y desviación típica 0.3, después truncamos para que los valores estén en [0,1]
            Wmod[pos] += rng.normal(scale=0.3)
            if Wmod[pos] < 0:
                Wmod[pos] = 0.0
            elif Wmod[pos] > 1:
                Wmod[pos] = 1.0

            vecinos += 1

            # Calculamos el fitness de la solución mutada
            fitmod = fitness(tasaClasificacion(Xtrain, Xtrain, Ytrain, Ytrain, Wmod), tasaReduccion(Wmod))
            eval += 1

            # En las transparencias se quiere minimizar, pero en este caso queremos maximizar, por lo que cambiamos el signo
            dif_fit = fit - fitmod

            # Si la solución mutada es mejor que la solución actual, la solución actual pasa a ser la mutada
            if dif_fit < 0:
                W = Wmod
                fit = fitmod
                exitos += 1
            # Si no lo es, todavía hay una probabilidad de que la solución actual pase a ser la mutada
            elif rng.random() < np.exp(-dif_fit / temp):
                W = Wmod
                fit = fitmod
                exitos += 1


        # Esquema de enfriamiento: esquema de Cauchy modificado
        beta = (temp - tempFinal) / (M * temp * tempFinal)
        temp = temp / (1 + beta * temp)

        # Si el número de éxitos en el enfriamiento actual es 0, paramos
        if exitos == 0:
            break


    return W, fit


# Implementación del algoritmo de búsqueda multiarranque básica
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def BMB(Xtrain, Ytrain, Xtest, Ytest):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    num_iter = 15
    num_eval = 1000

    # Generamos num_iter soluciones aleatorias y le aplicamos BL a cada una de ellas
    W_iniciales = rng.random((num_iter, num_caract))
    resultadosBL = np.array([busquedaLocalAux(Xtrain, Ytrain, W_iniciales[i], num_eval) for i in range(num_iter)], dtype=object)
    W_calculados, fitness_calculados = resultadosBL[:, 0], resultadosBL[:, 1]

    # Obtenemos la mejor solución y nos quedamos con esta
    pos = np.argmax(fitness_calculados)
    W = W_calculados[pos]

    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)
    

    return W, fit, tasa_clas, tasa_red, elapsed


# Implementación del algoritmo de ILS
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def ILS(Xtrain, Ytrain, Xtest, Ytest):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    tope = 1000
    iteraciones = 14    # Son 15 BL en total, contando la primera antes del bucle
    t = 0.1 * num_caract

    # Siempre se cambian al menos 2 características
    if t < 2:
        t = 2
    
    # Generamos una solución aleatoria y le aplicamos BL
    W = rng.random(num_caract)
    W, fit = busquedaLocalAux(Xtrain, Ytrain, W, tope)

    for i in range(iteraciones):
        # Hacemos una mutación fuerte de t componentes aleatorias
        posicionesAmutar = rng.choice(num_caract, int(t), replace=False)
        Wmod = W.copy()
        Wmod[posicionesAmutar] = rng.random(int(t))

        # Aplicamos BL a la solución mutada
        Wmod, fitmod = busquedaLocalAux(Xtrain, Ytrain, Wmod, tope)

        # Nos quedamos con la mejor solución
        if fitmod > fit:
            W = Wmod
            fit = fitmod

    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)

    return W, fit, tasa_clas, tasa_red, elapsed


# Implementación del algoritmo de ILS-ES
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def ILS_ES(Xtrain, Ytrain, Xtest, Ytest):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    tope = 1000
    iteraciones = 14    # Son 15 BL en total, contando la primera antes del bucle
    t = 0.1 * num_caract

    # Siempre se cambian al menos 2 características
    if t < 2:
        t = 2
    
    # Generamos una solución aleatoria y le aplicamos ES
    W = rng.random(num_caract)
    # También hace falta tope para ES???????????????
    W, fit = ESAux(Xtrain, Ytrain, W, tope)

    for i in range(iteraciones):
        # Hacemos una mutación fuerte de t componentes aleatorias
        posicionesAmutar = rng.choice(num_caract, int(t), replace=False)
        Wmod = W.copy()
        Wmod[posicionesAmutar] = rng.random(int(t))

        # Aplicamos ES a la solución mutada
        Wmod, fitmod = ESAux(Xtrain, Ytrain, Wmod, tope)

        # Nos quedamos con la mejor solución
        if fitmod > fit:
            W = Wmod
            fit = fitmod

    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)

    return W, fit, tasa_clas, tasa_red, elapsed


# Implementación del algoritmo de VNS
# param Xtrain = datos entrenamiento
# param Ytrain = clases entrenamiento
# param Xtest = datos test
# param Ytest = clases test
# return W = pesos, fit = fitness, tasa_clas = tasa de clasificación, tasa_red = tasa de reducción
def VNS(Xtrain, Ytrain, Xtest, Ytest):
    start = time.process_time()

    num_caract = Xtrain.shape[1]
    Kmax = 3
    K = 1
    tope = 1000
    iteraciones = 14    # Son 15 BL en total, contando la primera antes del bucle
    t = 0.1 * num_caract

    # Siempre se cambian al menos 2 características
    if t < 2:
        t = 2

    # Generamos una solución aleatoria y le aplicamos BL
    W = rng.random(num_caract)
    W, fit = busquedaLocalAux(Xtrain, Ytrain, W, tope, K)

    for i in range(iteraciones):
        # Hacemos una mutación fuerte de t componentes aleatorias
        posicionesAmutar = rng.choice(num_caract, int(t), replace=False)
        Wmod = W.copy()
        Wmod[posicionesAmutar] = rng.random(int(t))

        # Aplicamos BL a la solución mutada
        Wmod, fitmod = busquedaLocalAux(Xtrain, Ytrain, Wmod, tope, K)

        # Nos quedamos con la mejor solución
        if fitmod > fit:
            W = Wmod
            fit = fitmod
            # Si se mejora, volvemos a K=1
            K = 1
        # Si no, probamos con otra vecindad
        else:
            K += 1
            if K > Kmax:
                K = 1

    
    elapsed = time.process_time() - start

    tasa_clas = tasaClasificacion(Xtrain, Xtest, Ytrain, Ytest, W)
    tasa_red = tasaReduccion(W)
    fit = fitness(tasa_clas, tasa_red)

    return W, fit, tasa_clas, tasa_red, elapsed






if __name__ == "__main__":
    files = ["../BIN/diabetes_", "../BIN/ozone-320_", "../BIN/spectf-heart_"]
    titulos = ["Diabetes", "Ozone", "Spectf-heart"]
    algoritmos = ["BL", "ES", "BMB", "ILS", "ILS-ES", "VNS"]
    resultados = [[], [], [], [], [], []]
    
    seed = int(input("Introduce la semilla para generar números aleatorios: "))
    rng = np.random.default_rng(seed)



    # Para cada conjunto de datos
    for i in range(3):
        tablaBL = []
        tablaES = []
        tablaBMB = []
        tablaILS = []
        tablaILS_ES = []
        tablaVNS = []

        # Para cada partición
        for j in range(1,6):
            data = []
            test = readARFF(files[i] + str(j) + ".arff")

            # Usamos el método leave-one-out
            for k in range(1,6):
                if j != k:
                    data += readARFF(files[i] + str(k) + ".arff")

            # Separamos los datos de las clases
            Xtrain = []
            Ytrain = []

            for d in data:
                Xtrain.append(d[:-1])
                Ytrain.append(d[-1])

            # Repetimos el proceso con los datos de test
            Xtest = []
            Ytest = []

            for t in test:
                Xtest.append(t[:-1])
                Ytest.append(t[-1])

            # Normalizamos los datos (juntamos los datos de entrenamiento y test para normalizarlos juntos)
            X = np.array(Xtrain + Xtest)
            X = normalize(X)
            Xtrain = X[:len(Xtrain)]
            Xtest = X[len(Xtrain):]

            # Ejecutamos los algoritmos pedidos
            Wbl, fitness_bl, tasa_clas_bl, tasa_red_bl, elapsed_bl = busquedaLocal(Xtrain, Ytrain, Xtest, Ytest)
            Wes, fitness_es, tasa_clas_es, tasa_red_es, elapsed_es = ES(Xtrain, Ytrain, Xtest, Ytest)
            Wbmb, fitness_bmb, tasa_clas_bmb, tasa_red_bmb, elapsed_bmb = BMB(Xtrain, Ytrain, Xtest, Ytest)
            Wils, fitness_ils, tasa_clas_ils, tasa_red_ils, elapsed_ils = ILS(Xtrain, Ytrain, Xtest, Ytest)
            Wils_es, fitness_ils_es, tasa_clas_ils_es, tasa_red_ils_es, elapsed_ils_es = ILS_ES(Xtrain, Ytrain, Xtest, Ytest)
            Wvns, fitness_vns, tasa_clas_vns, tasa_red_vns, elapsed_vns = VNS(Xtrain, Ytrain, Xtest, Ytest)


            # Guardamos los resultados
            tablaBL.append([tasa_clas_bl, tasa_red_bl, fitness_bl, elapsed_bl, Wbl])
            tablaES.append([tasa_clas_es, tasa_red_es, fitness_es, elapsed_es, Wes])
            tablaBMB.append([tasa_clas_bmb, tasa_red_bmb, fitness_bmb, elapsed_bmb, Wbmb])
            tablaILS.append([tasa_clas_ils, tasa_red_ils, fitness_ils, elapsed_ils, Wils])
            tablaILS_ES.append([tasa_clas_ils_es, tasa_red_ils_es, fitness_ils_es, elapsed_ils_es, Wils_es])
            tablaVNS.append([tasa_clas_vns, tasa_red_vns, fitness_vns, elapsed_vns, Wvns])


        # Guardamos los resultados de cada conjunto de datos
        resultados[i].append(tablaBL)
        resultados[i].append(tablaES)
        resultados[i].append(tablaBMB)
        resultados[i].append(tablaILS)
        resultados[i].append(tablaILS_ES)
        resultados[i].append(tablaVNS)



    # Escribimos los resultados en un fichero csv
    # Para cada conjunto de datos
    for i in range(3):
        # Para cada algoritmo
        for j,alg in enumerate(algoritmos):
            with open("../../RESULTADOS/resultados_" + alg + "_" + titulos[i] + ".csv", "w") as f:
                writer = csv.writer(f)
                # Cabecera de la tabla
                writer.writerow(["", "%_clas", "%_red", "Fit.", "T"])
                # Datos de cada partición
                for k in range(5):
                    writer.writerow(["Partición " + str(k+1)] + [formatNumber(resultados[i][j][k][l]) for l in range(4)])
                    
                    # Imprimimos los pesos por pantalla
                    print("Pesos de la partición " + str(k+1) + " del algoritmo " + alg + " con el conjunto de datos " + titulos[i] + ":" + "\n" + str(resultados[i][j][k][4]) + "\n")
            

