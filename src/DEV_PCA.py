# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 22:42:34 2023

@author: ObregonW
"""

from sklearn.decomposition import PCA
import numpy as np

# Datos de ejemplo
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Crear una instancia de PCA
pca = PCA(n_components=2)  # Establecer el número de componentes principales deseados

# Aplicar PCA a los datos
data_pca = pca.fit_transform(data)


# Imprimir los resultados
print("Datos originales:\n", data)
print("Datos transformados por PCA:\n", data_pca)
