# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:33:02 2023

@author: ivoto
"""

#Regresion Lineal Multiple

#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 

dataset = pd.read_csv('50_Startups.csv')
#Definimos variable independientes, [filas, columnas (:-1 por que quiero todas menos la ultima)]
X = dataset.iloc[:, :-1].values 
#Definimos y(minuscula ya que es un vector en vez de una matriz) variable dependiente 
y = dataset.iloc[:, 4].values 


#Codificamos datos categoricos

#Codificar(pasar a numeros) datos categoricos(Es decir de categorias)
#Cabiamos por ej los pais a un numero correspondiente, como France = 1, Spain = 2 
#Y asi hasta tener todas las categorias ya codificadas
#Usamos la libreria sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Creamo dos, uno para los paises y otro para la matriz de caracteristicas(y)
labelencoder_X = LabelEncoder() #El constructor no necesita nada 
labelencoder_X.fit_transform(X[:,0])  #Me devuelve un array, de la codifcicaion 
#Ahora le doy el valor del encouder 
X[:,3] = labelencoder_X.fit_transform(X[:,3])
#Ahora ese valor es categorico por lo que tenemos que trasnformar a variable Dummy(Dummyficacion)
#Lo hacemos utilizando la misma libreria de skleanr y la transformacion que hicimos en el paso anterior
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)


#Evitar la trampa de las variables ficticias
#Podemos eliminar cualquiera, pero en este caso eliminamos la primera, ya que asi nos queda comodamente empezar de 1 y no de 0 (Pero no tiene que ser necesariamente asi)
X = X[:, 1:] #Selecciono todas las filas y desde la fila en adelantes (basicamente estoy eliminando la culumna cero)


#Training & Test 

#Dividir el dataset en conjunto de entrenamiento y de testing
#Utilizamos un libreria sklearn, model_selection (muy utilizidad para cross-validation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )



#Escalado de variables
'''
#Como la distancia euclidea (pitagoras de las variables ) puede tomar datos muy grande de una de las variables, lo que se hace es normalizar 
#Esto se hace para que un variable de valor muy grande no domine sobre el resto
#Escalar los datos = Normalizar los datos (Escalar a 0 y -1 correspode que el val max es 1 y el valor min es -1 ) 
#Hay dos metodos Standardisation(Campara de Gauss) y Normalisation(0 a 1 Lineal)
from sklearn.preprocessing import StandardScaler
#Escalamos el conjunto de entrenamiento 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
#El conjunto de test tiene que escalar con la misma tranformacion, no podemos usar una distinta para el conj de test 
X_test = sc_X.transform(X_test) #Solo detecta la transformacion y la aplica
#Ahora las variables de y-train e y_test no lo hacemos ya que es de clasificaion, por lo que no normalizamos 
#Si utilizaramos un algoritmo de prediccion(regresion lineal) hay que normalizar la y_train
'''

#Ajustar el modelo de Regresion lineal multiple con el conjunto de entrenamiento  

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

#Prediccion de los resultados en el conjunto de testing

y_pred = regression.predict(X_test)

#Construir el modelo optimo de RLM utilizando la Eliminacion hacia atras 

import statsmodels.api as sm
#No tenemos manera de ver el valor de la ordenada al origen, por lo que agregamos un columna al principio (haciendo referecia a Coeficiente cero (O.O.))
#Por loq ue agregamos una columna de 1 al principio que correspone con el termino independiente y podemos medir su P-Valor 
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()













































