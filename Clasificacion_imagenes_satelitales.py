# -*- coding: utf-8 -*-

#importar la biblioteca drive en la notebook
from google.colab import drive
drive.mount('/content/drive')

import sys, os
ruta_base = '/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Teledeteccion'
ruta_salida = '/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Teledeteccion/Imagenes_satelitales/Salidas'
ruta_shp = '/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Teledeteccion/Vectores'

"""## Librería Geopandas y Rasterio
**Geopandas** es una extension de la librería Pandas que nos permite trabajar con datos vectoriales.   En el **modelo vectorial** los datos adoptan la representación de elementos geométricos básicos como los puntos, las líneas o los polígonos. El elemento vectorial base es el **punto**; varios puntos unidos por segmentos definen una **línea**; y varias líneas y puntos se usan para definir **polígonos**. Geopandas toma elementos de otras librerías especialiazadas como shapely para realizar operaciones geométricas, y fiona para acceder y/o modificar los archivos. Es decir, se trata de un paquete de funcionalidades de alto nivel.  
**Rasterio** es un módulo de procesamiento de Python que está basado principalmente en GDAL y , por lo tanto, puede leer y escribir en todos los formatos de raster que ofrece GDAL. Ofrece algunas ventajas en cuanto a funcionalidades que ahorran mucho trabajo al usuario, puede ser mucho más amigable logrando los mismos resultados.
"""

# En esta notebook utilizaremos rasterio y geopandas
!pip3 install rasterio

## cosas que teniamos de antes ya:

import numpy as np
import matplotlib.pyplot as plt
import os

import rasterio
import rasterio.mask


from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask

import geopandas as gpd
from shapely.geometry import mapping

import pandas as pd
import seaborn as sns

# Leemos la imagen satelital
raster_fn = os.path.join(ruta_base, 'Imagenes_satelitales', 'Clasif_cultivos', 'Sentinel2_20230318_Zona4_UTM20S_clip.tif')
print(raster_fn)
with rasterio.open(raster_fn) as src:
    img = src.read()
    gt = src.transform
    band_names = src.descriptions

num_bandas = src.count
print(num_bandas)

"""# Definición de funciones para plotear la imagen, haciendo escalamiento y combinación de bandas"""

def scale(array,p = 0, nodata = None):
    '''
    Esta función escala o estira la imagen a determinado % del histograma (trabaja con percentiles)
    Si p = 0 (valor por defecto) entonces toma el mínimo y máximo de la imagen.
    Devuelve un arreglo nuevo, escalado de 0 a 1
    '''
    a = array.copy()
    a_min, a_max = np.percentile(a[a!=nodata],p), np.percentile(a[a!=nodata],100-p)
    a[a<a_min]=a_min
    a[a>a_max]=a_max
    return ((a - a_min)/(a_max - a_min))

# Función para realizar la combinación y el estiramiento de bandas de la imagen
def get_rgb(array,band_list, p = 0, nodata = None):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, una lista de índices correspondientes
    a las bandas que queremos usar, en el orden que deben estar (ej: [1,2,3]), y un parámetro
    p que es opcional, y por defecto es 0 (es el estiramiento a aplicar cuando llama a scale()).

    Devuelve una matriz con las 3 bandas escaladas

    Nota: Se espera una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    r = band_list[0]
    g = band_list[1]
    b = band_list[2]


    r1 = scale(array[r-1,:,:],p, nodata)
    g1 = scale(array[g-1,:,:],p, nodata)
    b1 = scale(array[b-1,:,:],p, nodata)


    a = np.dstack((r1,g1,b1))
    return a

def plot_rgb(array,band_list, p = 0, nodata = None, figsize = (12,6)):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, una lista de índices correspondientes
    a las bandas que queremos usar, en el orden que deben estar (ej: [1,2,3]), y un parámetro
    p que es opcional, y por defecto es 0 (es el estiramiento a aplicar cuando llama a get_rgb(), que a su vez llama a scale()).

    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.

    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: Se espera una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    r = band_list[0]
    g = band_list[1]
    b = band_list[2]

    a = get_rgb(array, band_list, p, nodata)

    plt.figure(figsize = figsize)
    plt.title(f'Combinación {r}, {g}, {b} \n (estirado al {p}%)' , size = 12)
    plt.imshow(a)
    plt.show()

plot_rgb(img, [3,2,1], p = 2)

# Leemos la capa de polígonos "ROIs"
rois_shp = gpd.read_file(ruta_shp+'/Rois_muestras.shp')
rois_shp.head(10)

print(src.crs)
print(rois_shp.crs)

"""¡Bien! Las capas tienen el mismo CRS. Ahora, para que podamos superponer las dos fuentes de información en el mismo gráfico, debemos explicitar cuál será la extensión a graficar con el parámetro extent. Para ello, primero usaremos una función de Rasterio que nos permite obtener esos límites:"""

from rasterio.plot import plotting_extent
s2_plot_extent = plotting_extent(src)

s2_plot_extent

rgb = get_rgb(img, [7,9,3], p=2)
fig, ax = plt.subplots(figsize = (12,6))
ax.imshow(rgb, extent = s2_plot_extent)
rois_shp.boundary.plot(ax=ax, edgecolor = 'black')
plt.title("Combinación bandas S2: 7-9-3")
plt.show()

"""## Uso de _mask_ para extraer valores de un raster a partir de un ROI

## Armo un dataframe con los datos etiquetados
### 1. Leo el shp como geoDataFrame y veo las clases
"""

#numero las clases de los ROIs alfabéticamente
clases=list(set(rois_shp['descrip']))
clases.sort()
clase_dict = {clase:i for i, clase in enumerate(clases)}
print(clase_dict)

"""### 2. Armo dos arrays X y Y con datos y etiquetas"""

#Leo los ROIS
with rasterio.open(raster_fn) as src:
    d=src.count #cantidad de bandas en el raster

nodata= -255 #elijo un valor raro para nodata

clases=list(set(rois_shp['clase']))
clases.sort() #numero las clases de los ROIs alfabéticamente
clase_dict = {clase:i for i, clase in enumerate(clases)}

#Preparo colección de atributos etiquetados. Comienza con 0 datos
X = np.zeros([0,d],dtype = np.float32) #array con todos los atributos
Y = np.zeros([0],dtype=int)            #array con sus etiquetas

with rasterio.open(raster_fn) as src:
    for index, row in rois_shp.iterrows():
        geom_sh = row['geometry']
        clase = row['clase']
        geom_GJ = [mapping(geom_sh)]
        clip, _transform = mask(src, geom_GJ, crop=True,nodata=nodata)
        d,x,y = clip.shape
        D = list(clip.reshape([d,x*y]).T)
        D = [p for p in D if (not (p==nodata).prod())]
        DX = np.array(D)
        DY = np.repeat(clase_dict[clase],len(D))
        X = np.concatenate((X,DX))
        Y = np.concatenate((Y,DY))
print(clases)

"""### Así X tiene una fila por cada pixel (con sus datos espectrales) e Y tiene un elemento por cada pixel indicando la clase correspondiente."""

# Previsualizar algunos datos etiquetados
print(X.shape)
print(np.unique(Y))
df_X = pd.DataFrame(i for i in X)
df_Y = pd.DataFrame(Y)

print(df_X.shape, 'dfX_shape')
print(df_Y.shape, 'dfY_shape')
datos = pd.concat([df_X, df_Y], axis = 1)

nombresColum = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
'MNDWI','NDBI', 'NDVI','EVI','REDNDVI', 'TVI',
'B8ASAVG', 'B8ACORR', 'B8ADVAR', 'B8ADISS', 'clase']
datos.columns = nombresColum
print(datos.head(5))
#print(datos.loc[datos['clase'] == 2])

"""## Firmas espectrales"""

def plot_firmas_espectrales(X, Y, B_G_R_NIR, clases, ylim=None, show_legend=True, title="Firmas espectrales de las clases", xlabel="Longitud de onda (nm)", ylabel="Reflectancia"):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    for j, l in enumerate(clases):
        S = np.where(Y == j)[0]
        data_vis_clase = X[S, :]  # Usa solo las columnas especificadas
        media_clase = data_vis_clase.mean(axis=0)
        std_clase = data_vis_clase.std(axis=0)
        print(media_clase)
        ci = std_clase
        ax.plot(B_G_R_NIR, media_clase, label=l)
        ax.fill_between(B_G_R_NIR, media_clase - ci, media_clase + ci, color='b', alpha=.1)

    if ylim:
        plt.ylim(ylim)

    if show_legend:
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    font = {'family': 'serif', 'color': 'darkgray', 'size': 12}
    for x, t in zip(B_G_R_NIR, ['Blue', 'Green', 'Red', 'NIR']):
        plt.text(x - 0.012, 0.03, t, fontdict=font)

    plt.show()

X_selected = X[:, [0, 1, 2, 6]] #Bandas de X
B_G_R_NIR = [0.49,0.56, 0.665, 0.842]
print(X_selected[5])
#clases_plot = np.unique(Y)
clases_plot = [0, 1, 2, 5] # Clases o categorías ('agua': 0, 'alfalfa': 1, 'algodon': 2, 'areas_inundadas': 5)
print(clases_plot)
plot_firmas_espectrales(X_selected, Y, B_G_R_NIR, clases_plot, ylim=[0, 5000])

"""## Boxplots"""

#Armo un dataframe con los datos etiquetados
data=pd.DataFrame({'B':X[:,0],'G':X[:,1],'R':X[:,2],'NIR':X[:,6],'NDVI':X[:,12],'clase':[clases[y] for y in Y]})

#y los grafico de diferentes formas
fig = plt.figure(figsize=(15,10))
sns.boxplot(data=data,x='clase', y='NDVI')
plt.show()

"""**La clasificación es una tarea supervisada** y, ya que estamos interesados en su rendimiento en datos no utilizados para entrenar, vamos a dividir los datos en dos partes:

un conjunto de entrenamiento que el algoritmo de aprendizaje utiliza para ajustar los parámetros del modelo
un conjunto de test para evaluar la capacidad de generalización del modelo
La función train_test_split del paquete model_selection hace justo esto por nosotros - la usaremos para generar una partición con un 75% y 25% en entrenamiento y test, respectivamente.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score as kappa
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        datos.drop(columns = ["clase"]),
                                        datos['clase'],
                                        stratify= datos['clase'],
                                        random_state = 123
                                    )

"""**Verificar que la distribución de la variable respuesta es similar en el conjunto de entrenamiento y en el de test.**"""

print("Partición de entrenamento")
print("-----------------------")
print(y_train.describe())
print("Partición de test")
print("-----------------------")
print(y_test.describe())

numeric_cols = X_train.select_dtypes(include=['float32', 'int32']).columns.to_list()
preprocessor = ColumnTransformer(
                   [('scale', StandardScaler(), numeric_cols)],
                remainder='passthrough')
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

clf = RandomForestClassifier(criterion='entropy', max_features=10, n_estimators=500, random_state=123)
clf = clf.fit(X_train_prep, y_train)
Y_pred = clf.predict(X_test_prep)

print("Confusion Matrix:")
print(confusion_matrix(y_test, Y_pred))

print("Classification Report")
print(classification_report(y_test, Y_pred))

"""# Importancia de las características
La importancia de las variables en un Random Forest de scikit-learn se basa en la disminución de impureza que cada característica aporta a lo largo de los árboles del bosque.
# Incremento de la pureza de nodos

Cuantifica el incremento total en la pureza de los nodos debido a divisiones en las que participa el predictor (promedio de todos los árboles). La forma de calcularlo es la siguiente: en cada división de los árboles, se registra el descenso conseguido en la medida empleada como criterio de división (índice Gini, mse, entropía, ...). Para cada uno de los predictores, se calcula el descenso medio conseguido en el conjunto de árboles que forman el ensemble. Cuanto mayor sea este valor medio, mayor la contribución del predictor en el modelo. **[Más info](https://cienciadedatos.net/documentos/py08_random_forest_python)**
"""

# making a pandas dataframe
data = list(zip(X_train.columns, clf.feature_importances_))
data
df_importances = pd.DataFrame(data, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
df_importances

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# Creating a bar plot
fig, ax = plt.subplots(figsize=(7, 8))
df_importances = df_importances.sort_values('Importance', ascending=False)
sns.barplot(x=df_importances['Importance'], y=df_importances['Feature'])
# Add labels to your graph
plt.xlabel('Importancia de los predictores')
plt.ylabel('Predictores')
plt.title("Importancia de los predictores")
plt.legend()
plt.show()

"""## 2. Ejercicio de clasificación
1. Ajustar un nuevo modelo que considere las variables de mayor importancia.
    1. Obtener la matriz de confusión.
    1. Comparar el desempeño del modelo con todas las variables versus el modelo con las variables seleccionadas.  
    

"""

datos=datos.drop(columns = ["B8ACORR",'B8ADISS','B8ADVAR','TVI'])

# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        datos.drop(columns = ["clase"]),
                                        datos['clase'],
                                        stratify= datos['clase'],
                                        random_state = 123
                                    )

numeric_cols = X_train.select_dtypes(include=['float32', 'int32']).columns.to_list()
preprocessor = ColumnTransformer(
                   [('scale', StandardScaler(), numeric_cols)],
                remainder='passthrough')
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

clf = RandomForestClassifier(criterion='entropy', max_features=10, n_estimators=500, random_state=123)
clf = clf.fit(X_train_prep, y_train)
Y_pred = clf.predict(X_test_prep)

print("Confusion Matrix:")
print(confusion_matrix(y_test, Y_pred))

print("Classification Report")
print(classification_report(y_test, Y_pred))

"""###El rendimiento del modelo es muy similar al anterior eliminando las predictoras "B8ACORR",'B8ADISS','B8ADVAR','TVI' la principal perdida que se ve es la disminucion en el recall de 0.97 a 0.90 para la categoria 6 maleza."""

# making a pandas dataframe
data = list(zip(X_train.columns, clf.feature_importances_))
data
df_importances = pd.DataFrame(data, columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
df_importances

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# Creating a bar plot
fig, ax = plt.subplots(figsize=(7, 8))
df_importances = df_importances.sort_values('Importance', ascending=False)
sns.barplot(x=df_importances['Importance'], y=df_importances['Feature'])
# Add labels to your graph
plt.xlabel('Importancia de los predictores')
plt.ylabel('Predictores')
plt.title("Importancia de los predictores")
plt.legend()
plt.show()