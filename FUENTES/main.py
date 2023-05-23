import numpy as np
import time

# Semilla para la generación de números aleatorios
rng = np.random.default_rng(0)

# Función auxiliar para convertir los string que sean números a float
def toFloat(x):
    try:
        return float(x)
    except ValueError:
        return x
    

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
# param W = pesos
# return W = pesos, fit = fitness, eval = evaluaciones
def busquedaLocal(Xtrain, Ytrain, W, fit):
    num_caract = Xtrain.shape[1]
    mutadas = np.ones(num_caract)
    eval = 0
    generados = 0
    iteraciones = 2.0 * W.size


    while eval < iteraciones:
        # Mutamos una componente aleatoria sin repetición (emulamos un bucle do-while)
        while True:
            pos = rng.integers(num_caract)

            if mutadas[pos] == 1:
                mutadas[pos] = 0
                break

        Wmod = W.copy()
        # Usamos una mutación normal con media 0 y desviación típica 0.3, después truncamos para que el valor esté en [0,1]
        Wmod[pos] += rng.normal(scale=0.3)
        Wmod = np.clip(Wmod, 0, 1)

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



    return W, fit, eval





if __name__ == "__main__":
    files = ["../BIN/diabetes_", "../BIN/ozone-320_", "../BIN/spectf-heart_"]
    titulos = ["Diabetes", "Ozone", "Spectf-heart"]
    algoritmos = ["AGG-BLX", "AGG-CA", "AGE-BLX", "AGE-CA", "AM-(10,1.0)", "AM-(10,0.1)", "AM-(10,0.1mej)"]
    resultados = [[], [], [], [], [], [], []]
    
    seed = int(input("Introduce la semilla para generar números aleatorios: "))
    rng = np.random.default_rng(seed)



    # Para cada conjunto de datos
    for i in range(3):
        tablaAggBlx = []
        tablaAggArit = []
        tablaAgeBlx = []
        tablaAgeArit = []
        tablaAmAll = []
        tablaAmRand = []
        tablaAmBest = []
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

            # Ejecutamos los algoritmos genéticos
            Wagg_blx, fitness_agg_blx, tasa_clas_agg_blx, tasa_red_agg_blx, elapsed_agg_blx = AGG_BLX(Xtrain, Ytrain, Xtest, Ytest)
            Wagg_arit, fitness_agg_arit, tasa_clas_agg_arit, tasa_red_agg_arit, elapsed_agg_arit = AGG_Arit(Xtrain, Ytrain, Xtest, Ytest)
            Wage_blx, fitness_age_blx, tasa_clas_age_blx, tasa_red_age_blx, elapsed_age_blx = AGE_BLX(Xtrain, Ytrain, Xtest, Ytest)
            Wage_arit, fitness_age_arit, tasa_clas_age_arit, tasa_red_age_arit, elapsed_age_arit = AGE_Arit(Xtrain, Ytrain, Xtest, Ytest)

            # Ejecutamos los algoritmos meméticos
            Wam_all, fitness_am_all, tasa_clas_am_all, tasa_red_am_all, elapsed_am_all = AM_All(Xtrain, Ytrain, Xtest, Ytest)
            Wam_rand, fitness_am_rand, tasa_clas_am_rand, tasa_red_am_rand, elapsed_am_rand = AM_Rand(Xtrain, Ytrain, Xtest, Ytest)
            Wam_best, fitness_am_best, tasa_clas_am_best, tasa_red_am_best, elapsed_am_best = AM_Best(Xtrain, Ytrain, Xtest, Ytest)

            # Guardamos los resultados
            tablaAggBlx.append([tasa_clas_agg_blx, tasa_red_agg_blx, fitness_agg_blx, elapsed_agg_blx, Wagg_blx])
            tablaAggArit.append([tasa_clas_agg_arit, tasa_red_agg_arit, fitness_agg_arit, elapsed_agg_arit, Wagg_arit])
            tablaAgeBlx.append([tasa_clas_age_blx, tasa_red_age_blx, fitness_age_blx, elapsed_age_blx, Wage_blx])
            tablaAgeArit.append([tasa_clas_age_arit, tasa_red_age_arit, fitness_age_arit, elapsed_age_arit, Wage_arit])
            tablaAmAll.append([tasa_clas_am_all, tasa_red_am_all, fitness_am_all, elapsed_am_all, Wam_all])
            tablaAmRand.append([tasa_clas_am_rand, tasa_red_am_rand, fitness_am_rand, elapsed_am_rand, Wam_rand])
            tablaAmBest.append([tasa_clas_am_best, tasa_red_am_best, fitness_am_best, elapsed_am_best, Wam_best])


        # Guardamos los resultados de cada conjunto de datos
        resultados[i].append(tablaAggBlx)
        resultados[i].append(tablaAggArit)
        resultados[i].append(tablaAgeBlx)
        resultados[i].append(tablaAgeArit)    
        resultados[i].append(tablaAmAll)
        resultados[i].append(tablaAmRand)
        resultados[i].append(tablaAmBest)



    # Imprimimos los resultados
    for i in range(3):
        print("=================================")
        print(titulos[i])
        print("=================================")

        for j in range(7):
            print("*****", algoritmos[j], "*****")
            print("\t\t %tasa_clas\t %tasa_red\t Fit.\t T")
            print("---------------------------------")

            for k in range(5):
                print("Partición ", str(k+1), "\t", resultados[i][j][k][0], "\t", resultados[i][j][k][1], "\t", resultados[i][j][k][2], "\t", resultados[i][j][k][3])
                print("W: ", resultados[i][j][k][4])

            print("\n")
            

