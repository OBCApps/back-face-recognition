# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 23:47:11 2023

@author: ObregonW
"""

from rtree import index

def build_rtree_index(data):
    p = index.Property()
    p.dimension = len(data[0])  # Dimensionality de los vectores característicos
    idx = index.Index(properties=p)
    
    for i, vector in enumerate(data):
        idx.insert(i, tuple(vector))
    
    return idx

def knn_rtree(query, idx, data, k):
    result = list(idx.nearest(tuple(query), k))
    return [(data[i], i) for i in result]

def knn_rtree_all(query, idx, data):
    result = list(idx.intersection(tuple(query)))
    return [(data[i], i) for i in result]

# Ejemplo de uso
data = [[1, 2, 3, 2, 3, 2, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3], 
        [4, 5, 6, 2, 3, 2, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3], 
        [7, 8, 9, 2, 3, 2, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3]]  # Vectores característicos de las imágenes

query = [2, 3, 4, 2, 3, 2, 3, 2, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3]  # Imagen de consulta

# Construir el índice R-tree
idx = build_rtree_index(data)

# Realizar una búsqueda KNN utilizando el índice R-tree
knn_results = knn_rtree(query, idx, data, k=3)

knn_results_all = knn_rtree_all(query, idx, data)

# Imprimir los resultados
print("PRIMERO")
for result in knn_results:
    print("Imagen:", result[0])
    print("Índice:", result[1])
    print()

print("SEGNDO")
for result in knn_results:
    print("Imagen:", result[0])
    print("Índice:", result[1])
    print()