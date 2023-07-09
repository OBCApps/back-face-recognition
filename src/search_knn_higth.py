from rtree import index
import os
from src.search_knn_secuencial import return_images
from sklearn.decomposition import PCA
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
dir_dataset = os.path.join(current_dir, "dataset", "images_local1")
dir_input = os.path.join(current_dir, "input")

def build_rtree_index_PCA(data):
    #print(data)
    p = index.Property()
    p.dimension = 128
    
    idx = index.Index(properties=p)
    for name, characteristic in data.items():
        for i, vector in enumerate(characteristic , start=1):
            vector = tuple(vector)
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(vector)
            idx.insert(i, data_pca , obj=(name, i))
    return idx


def buscar_indice(idx, query):
    resultados = idx.intersection(query, objects=True)
    k_elementos_cercanos = []
    
    for resultado in resultados:
        name, index_image = resultado.object
        index_str = str(index_image).zfill(4)  
        
        k_elementos_cercanos.append((name, index_str))
    return k_elementos_cercanos


def buscar_knn_rtree(idx, query, k):
    resultados = idx.nearest(query, num_results=k, objects=True)
    k_elementos_cercanos = []
    
    for resultado in resultados:
        name, index_image = resultado.object
        index_str = str(index_image).zfill(4)  
        
        k_elementos_cercanos.append((name, index_str))
    
    return k_elementos_cercanos


# MIO
def search_rtree_indexed_all(query, data):
    print("search_rtree_indexed_all ")
    idx = build_rtree_index_PCA(data)
    
    print("TOTAL CONSTRUIDOS: " , len(idx))
    start_time = time.time() 
    resultados = buscar_indice(idx , query)
    execution_time = time.time() - start_time  
    return execution_time, return_images(resultados)

def search_rtree_indexed_knn(query, data, k):
    print("search_rtree_indexed_knn ")
    idx = build_rtree_index_PCA(data)
    print("TOTAL CONSTRUIDOS: " , len(idx))
    
    start_time = time.time() 
    resultados = buscar_knn_rtree(idx , query, k)
    execution_time = time.time() - start_time  
    return execution_time, return_images(resultados)
