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

# Obtener los componentes principales
componentes_principales = pca.components_

# Obtener la varianza explicada por cada componente
varianza_explicada = pca.explained_variance_ratio_

# Obtener la varianza total explicada por los componentes principales
varianza_total_explicada = np.sum(varianza_explicada)

# Imprimir los resultados
print("Datos originales:\n", data)
print("Datos transformados por PCA:\n", data_pca)
print("Componentes principales:\n", componentes_principales)
print("Varianza explicada por cada componente:\n", varianza_explicada)
print("Varianza total explicada por los componentes principales:", varianza_total_explicada)
