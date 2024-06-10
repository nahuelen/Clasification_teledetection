# -*- coding: utf-8 -*-


#importar la biblioteca drive en la notebook
from google.colab import drive
drive.mount('/content/drive')

import sys, os
ruta_base = '/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Teledeteccion/Imagenes_satelitales/Sentinel2_mar24.tif'

###Iporto librerias
import numpy as np
from osgeo import gdal

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

ruta = os.path.join('/content/drive/MyDrive/Uba/Maestria datos financieros/Metodos datos no estructurados/Teledeteccion/Imagenes_satelitales/Sentinel2_mar24.tif')
print(ruta)
img_sentinel2 = gdal.Open(ruta)

print(img_sentinel2)

gt = img_sentinel2.GetGeoTransform()
src = img_sentinel2.GetProjection()
num_bandas = img_sentinel2.RasterCount
print(num_bandas)

# Obtener los nombres de las bandas de la imagen sentinel 2
nombres_bandas = []
for i in range(1, num_bandas + 1):
    banda = img_sentinel2.GetRasterBand(i)
    nombres_bandas.append(banda.GetDescription() or f'Banda {i}')
print("Nombres de las bandas:", nombres_bandas)

img_sentinel2_array = img_sentinel2.ReadAsArray()##transformo a array

dim = img_sentinel2_array.shape##veo dimensiones de la imagen 8 bandas por 2783 px de alto y 4504 px de ancho
print(dim)

banda11 = img_sentinel2_array[6,:,:] ##visualizo banda 11 en orden 7
plt.figure(figsize = (12,6))
plt.imshow(banda11, cmap = 'gray')
plt.title("Banda 11 ")
plt.show()

# Leamos algunas estadísticas de cada banda
#Para visualizar cada banda, utilizamos el slicing sobre los arreglos de numpy
# Se muestra el ejemplo para la banda 1. Completar para el resto de las bandas

banda1 = img_sentinel2_array[0,:,:]
banda2 = img_sentinel2_array[1,:,:]
banda3 = img_sentinel2_array[2,:,:]
banda4 = img_sentinel2_array[3,:,:]

#Mínimo y máximo:
min_b1, max_b1 = np.min(banda1), np.max(banda1)
min_b2, max_b2=np.min(banda2), np.max(banda2)
min_b3, max_b3=np.min(banda3), np.max(banda3)
min_b4, max_b4=np.min(banda4), np.max(banda4)
#Percentiles 2% y 98%
p2_b1 = np.percentile(banda1, 2)
p98_b1 = np.percentile(banda1, 98)
p2_b2 = np.percentile(banda2, 2)
p98_b2 = np.percentile(banda2, 98)
p2_b3 = np.percentile(banda3, 2)
p98_b3 = np.percentile(banda3, 98)
p2_b4 = np.percentile(banda4, 2)
p98_b4 = np.percentile(banda4, 98)
print(f'Mínimo:{min_b1}, Máximo:{max_b1}')
print(f'Percentil 2%: {p2_b1}, Percentil 98%: {p98_b1}')
print(f'Mínimo:{min_b2}, Máximo:{max_b2}')
print(f'Percentil 2%: {p2_b2}, Percentil 98%: {p98_b2}')
print(f'Mínimo:{min_b3}, Máximo:{max_b3}')
print(f'Percentil 2%: {p2_b3}, Percentil 98%: {p98_b3}')
print(f'Mínimo:{min_b4}, Máximo:{max_b4}')
print(f'Percentil 2%: {p2_b4}, Percentil 98%: {p98_b4}')

import matplotlib.pyplot as plt

# Crear una figura con subgráficos
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Gráfico 1: Histograma de banda1
axs[0, 0].hist(banda1.ravel(), bins=100)
axs[0, 0].axvline(p2_b1, color='red', linestyle='--', label='Percentil 2%')
axs[0, 0].axvline(p98_b1, color='black', linestyle='--', label='Percenil 98%')
axs[0, 0].legend()
axs[0, 0].set_title('Histograma Banda 1')

# Gráfico 2: Histograma de banda2
axs[0, 1].hist(banda2.ravel(), bins=100)
axs[0, 1].axvline(p2_b2, color='red', linestyle='--', label='Percentil 2%')
axs[0, 1].axvline(p98_b2, color='black', linestyle='--', label='Percenil 98%')
axs[0, 1].legend()
axs[0, 1].set_title('Histograma Banda 2')

# Gráfico 3: Histograma de banda3
axs[1, 0].hist(banda3.ravel(), bins=100)
axs[1, 0].axvline(p2_b3, color='red', linestyle='--', label='Percentil 2%')
axs[1, 0].axvline(p98_b3, color='black', linestyle='--', label='Percenil 98%')
axs[1, 0].legend()
axs[1, 0].set_title('Histograma Banda 3')

# Gráfico 4: Histograma de banda4
axs[1, 1].hist(banda4.ravel(), bins=100)
axs[1, 1].axvline(p2_b4, color='red', linestyle='--', label='Percentil 2%')
axs[1, 1].axvline(p98_b4, color='black', linestyle='--', label='Percenil 98%')
axs[1, 1].legend()
axs[1, 1].set_title('Histograma Banda 4')

# Ajustar el espaciado entre los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()

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

"""
Exploro otras escalas de colores para visualizar cada imagen (hasta ahora venimos usando la escala de grises). En este enlace se pueden consultar algunas de las rampas de color disponibles para MatplotLib.
"""

#Estirando al 10% - 90% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 1 - 10% / 90%")
plt.imshow(scale(banda1, p = 10), cmap = 'summer')
plt.show()

#Estirando al 20% - 80% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 1 - 20% / 80%")
plt.imshow(scale(banda1, p = 20), cmap = 'summer')
plt.show()

#Estirando al 10% - 90% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 2 - 10% / 90%")
plt.imshow(scale(banda2, p = 10), cmap = 'copper')
plt.show()

#Estirando al 20% - 80% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 2 - 20% / 80%")
plt.imshow(scale(banda2, p = 20), cmap = 'copper')
plt.show()

#Estirando al 10% - 90% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 3 - 10% / 90%")
plt.imshow(scale(banda3, p = 10), cmap = 'hot')
plt.show()

#Estirando al 20% - 80% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 3 - 20% / 80%")
plt.imshow(scale(banda3, p = 20), cmap = 'hot')
plt.show()

#Estirando al 10% - 90% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 4 - 10% / 90%")
plt.imshow(scale(banda4, p = 10), cmap = 'Purples')
plt.show()

#Estirando al 20% - 80% de la imagen
plt.figure(figsize = (12,6))
plt.title("Banda 4 - 20% / 80%")
plt.imshow(scale(banda4, p = 20), cmap = 'Purples')
plt.show()

#Elegimos el orden de las bandas con las que queremos trabajar
r = 3 #irá en el canal rojo
g = 2 # irá en el canal verde
b = 1 # irá en el canal azul

# Escalamos las bandas entre 0 y 1 para que sea más fácil su visualización
# Recordar las indexaciones en Python!!, por eso le restamos 1 a cada número de banda!

r1 = img_sentinel2_array[r-1,:,:]
g1 = img_sentinel2_array[g-1,:,:]
b1 = img_sentinel2_array[b-1,:,:]

a = np.dstack((r1,g1,b1))

plt.figure(figsize = (12,6))
plt.title(f'Combinación {r}, {g}, {b}', size = 10)
plt.imshow(a)
plt.show()

## SIN ESCALONAMIENTO QUEDA SATURADA
#Elegimos el orden de las bandas con las que queremos trabajar
r = 3 #irá en el canal rojo
g = 2 # irá en el canal verde
b = 1 # irá en el canal azul

# Escalamos las bandas entre 0 y 1 para que sea más fácil su visualización
# Recordar las indexaciones en Python!!, por eso le restamos 1 a cada número de banda!

r1 = img_sentinel2_array[r-1,:,:]
g1 = img_sentinel2_array[g-1,:,:]
b1 = img_sentinel2_array[b-1,:,:]

a = np.dstack((r1,g1,b1))

plt.figure(figsize = (12,6))
plt.title(f'Combinación {r}, {g}, {b}', size = 10)
plt.imshow(a)
plt.show()



# Escalamos las bandas entre percentiles del 5% y 95% para que sea más fácil su visualización
# Recordar las indexaciones en Python!!, por eso le restamos 1 a cada número de banda!

r1 = img_sentinel2_array[r-1,:,:]
g1 = img_sentinel2_array[g-1,:,:]
b1 = img_sentinel2_array[b-1,:,:]

r1=scale(r1,5)
g1=scale(g1,5)
b1=scale(b1,5)
a = np.dstack((r1,g1,b1))

plt.figure(figsize = (12,6))
plt.title(f'Combinación al 5% y 95% {r}, {g}, {b}', size = 10)
plt.imshow(a)

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

imagen=get_rgb(img_sentinel2_array,[1,2,3],10)

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
    plt.title(f'Combinación {r}, {g}, {b} \n (estirado al {p}%)' , size = 20)
    plt.imshow(a)
    plt.show()

plot_rgb(img_sentinel2_array,[1,2,3],10)

plot_rgb(img_sentinel2_array,[1,2,3],5)



sierras_guasayan = img_sentinel2_array[:,0:2500,3000:4000]
plot_rgb(sierras_guasayan,[1,2,3], 5)

banda1 = sierras_guasayan[0,:,:]
banda2 = sierras_guasayan[1,:,:]
banda3 = sierras_guasayan[2,:,:]

# Gráfico 1: Histograma de banda1
plt.figure(figsize = (12,6))
plt.hist(banda1.ravel(), bins=100)
plt.legend()
plt.title('Histograma Banda 1')
plt.show()
# Gráfico 2: Histograma de banda2
plt.figure(figsize = (12,6))
plt.hist(banda2.ravel(), bins=100)
plt.legend()
plt.title('Histograma Banda 2')
plt.show()
# Gráfico 3: Histograma de banda3
plt.figure(figsize = (12,6))
plt.hist(banda3.ravel(), bins=100)
plt.legend()
plt.title('Histograma Banda 3')
plt.show()

#Cálculo  y visualización del NDVI

# Tip: Recordar que se pueden consultar distintas paletas de color de matplotlib aquí: https://matplotlib.org/stable/tutorials/colors/colormaps.html

b_nir = img_sentinel2_array[3,:,:]
b_red = img_sentinel2_array[2,:,:]

ndvi = (b_nir - b_red) / (b_nir + b_red)
plt.figure(figsize = (12,6))
plt.imshow(ndvi, vmin = 0, vmax = 1, cmap = 'gray')
plt.title("NDVI")
plt.show()

plt.figure(figsize = (12,6))
plt.imshow(ndvi, vmin = 0, vmax = 1, cmap = 'terrain_r')
#plt.imshow(scale(ndvi,10), cmap = 'terrain_r')
plt.title("NDVI")
# Add colorbar to show the index
plt.colorbar()
plt.show()

### genero indice de verdosidad

band_3 = img_sentinel2_array[2,:,:]
band_11 = img_sentinel2_array[7,:,:]

ndwi = (band_3 - band_11) / (band_3 + band_11)
plt.figure(figsize = (12,6))
plt.imshow(ndwi, vmin = -1, vmax = 1, cmap = 'gray')
plt.title("NDWI")
plt.show()

plt.figure(figsize = (12,6))
plt.imshow(ndwi, cmap = 'terrain_r')
plt.title("NDWI")
# Add colorbar to show the index
plt.colorbar()
plt.show()

#SAVI (Sentinel 2) = (B08 – B04) / (B08 + B04 + 0,428) * (1,428)
band_4 = img_sentinel2_array[2,:,:]
band_8 = img_sentinel2_array[3,:,:]

SAVI = (band_8 - band_4) / (band_8 + band_4+ 0.428)* (1.428)
plt.figure(figsize = (12,6))
plt.imshow(SAVI, vmin = -1, vmax = 1, cmap = 'gray')
plt.title("SAVI")
plt.show()

plt.figure(figsize = (12,6))
plt.imshow(SAVI, cmap = 'terrain_r')
plt.title("SAVI")
# Add colorbar to show the index
plt.colorbar()
plt.show()

