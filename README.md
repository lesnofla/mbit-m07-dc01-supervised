# EJERCICIO SUPERVISED LEARNING (Python)
# Carlos Alfonsel (carlos.alfonsel@mbitschool.com)

# 1. Análisis Exploratorio del Dataset: Auditoría de Datos, Missing Values y Correlación de Variables.

En primer lugar, se realiza un preprocesado del dataset, para obtener un conjunto de datos lo más "limpio" posible para trabajar con él. Para ello, una vez importado el dataset a un dataframe, se revisa la estructura de los datos, se obtienen unas estadísticas básicas de las variables numéricas, se comprueba que no haya registros duplicados, etc.

**Missing Values**: se observa que un número elevado de variables tienen todos sus valores a NaN, por lo que se procede a eliminar dichas variables.

varsToBeErased = ["user_name", ..., "var_yaw_forearm"]
df_test1       = df_test1.drop(varsToBeErased, axis = 1)

**Matriz de Correlación**: una vez calculada la matriz de correlación entre las variables restantes, se observa que algunas muestran valores altos de correlación (> 0.75), por lo que se eliminan del dataset.

varsToBeErasedCorr = ["accel_belt_x", ... ,"gyros_dumbbell_x"]
df_test2           = df_test1.drop(varsToBeErasedCorr, axis = 1)

Una vez realizadas estas dos operaciones, ya tenemos nuestro dataset con 32 variables/predictores preparado para entrenarlo y validarlo con distintos modelos de Aprendizaje Supervisado.


# 2. Entrenamiento/Validación de Modelos de Aprendizaje Supervisado sobre el Dataset:

Antes de proceder a aplicar los distintos modelos, vamos a realizar una **Normalización** de los datos, ya que algunas variables presentan distintos órdenes de magnitud y eso puede afectar al rendimiento de alguno de los modelos a entrenar.

Para ello, utilizamos la función **MinMaxScaler()**:

X = df.iloc[:, range(0,32)]
Y = df.iloc[:, 32]

scaled_X = MinMaxScaler().fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns = X.columns)

A continuación, creamos los conjuntos de entrenamiento y validación:

X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size = 0.3, random_state = 1)

## 2.1. Árboles de Decisión: DecisionTreeClassifier()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.99 con este algoritmo.

## 2.2. Random Forest: RandomForestClassifier()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.99 con este algoritmo.

## 2.3. Gradient-Boosted Trees: GradientBoostingClassifier()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.99 con este algoritmo.

## 2.4. k-NN (k-Nearest Neighbors): KNeighborsClassifier()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.98 con este algoritmo.

## 2.5. SVM (Support Vector Machine): SVC()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.98 con este algoritmo.

## 2.6. Artificial Neural Network (ANN): MLPClassifier()

Se realiza un estudio del *overfitting* o *sobreajuste* entre los conjuntos de *training* y *test*, así como un ajuste de parámetros mediante la función **doGridSearch** para optimizar el modelo. Al final se consigue un **Accuracy** = 0.96 con este algoritmo.

## 2.7. Neural Networks con Keras y Tensorflow

Se define un modelo secuencial de red de neuronas con las siguientes características:

- Número de capas: 5
- Neuronas en la capa de entrada = número de predictores (32)
- Neuronas en la capa de salida  = número de clases de la variable target (5)
- Función de activación: relu

model = Sequential()
model.add(Dense(32, activation = 'relu', input_shape=(32,)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense( 5, activation = 'softmax'))

Con un número de *epochs* igual a 50, obtenemos un **Accuracy** = 0.9438 (94.4%) con el conjunto de *test*.

## NOTA: Gaussian y Multinomial Naive-Bayes

Se ha probado también con estos algoritmos, y al igual que con los de regresión logística y discrimincación lineal, se obtienen datos de precisión por debajo del 50%, por lo que se considera que no son adecuados para trabajar con este conjunto de datos en particular.


## 3. CROSS-VALIDATION ENTRE MODELOS DE APRENDIZAJE SUPERVISADO:

Utilizando un método de validación cruzada (cross_val_score), comparamos los modelos vistos anteriormente, obteniendo los siguientes resultados:

            Mean      Std
**CART**: 0.956886 (0.008685)
**RF**  : 0.995414 (0.004232)
**GBT** : 0.956019 (0.011125)
**KNN** : 0.969066 (0.008508)
**SVM** : 0.897358 (0.014556)
**MLP** : 0.918614 (0.013716)

## 4. RESUMEN Y CONCLUSIONES:

Se puede concluir que el modelo que mejores resultados obtiene con este conjunto de datos, entrenado y validado con 32 predictores, es **Random Forest**, con una precisión de **0.995** (muy cerca del 100% de *accuracy*). Por último, también se ha podido comprobar que un modelo de tipo Naive-Bayes no es adecuado para predecir el valor de la variable objetivo del dataset, o por lo menos con las funciones y parámetros que proporciona el paquete *sklearn* de Python.

Por último, se ha estudiado que con una red densa de neuronas de 5 capas, con 32 en la de entrada, 5 en la de salida y 64 neuronas en cada una de las tres capas internas, se obtienen también buenos resultados en términos de precisión, llegando a obtener un **0.9438**. La función de activación utilizada es *ReLU (Rectified Lineal Unit)*, el optimizador es de tipo *adam*, y el número de *epochs* es igual a 50.