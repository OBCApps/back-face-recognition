import uvicorn
from fastapi import FastAPI
import nest_asyncio
import os

from src.search_knn_secuencial import knn_secuencial_rango, knn_secuencial_pq ,return_images
from src.face_reconition import characterist_image, read_characteristics
from src.search_R_tree_index import search_rtree_indexed_all, search_rtree_indexed_knn

app = FastAPI()
nest_asyncio.apply()


current_dir = os.path.dirname(os.path.abspath(__file__))
directory_dataset = os.path.join(current_dir, "src", "index")
directory_input = os.path.join(current_dir, "src", "input")


# print(f"Main: Ruta actual = {current_dir}" )
# print(f"Main: directory_dataset = {directory_dataset}" )
# print(f"Main: directory_input = {directory_input}" )

@app.get("/search-secuencial-pq")
def read_root():    
    query = characterist_image(directory_input + '\\uno1.JPG')
    data = read_characteristics(directory_dataset)
    k = 4
    finds = knn_secuencial_pq(query, data, k)
    return return_images(finds)


@app.get("/search-secuencial-range")
def read_root1():        
    query = characterist_image(directory_input + '\\uno1.JPG')
    data = read_characteristics(directory_dataset)
    k = 4
    radio = 5
    finds = knn_secuencial_rango(query, data, radio, k)
    return return_images(finds)

@app.get("/search-rtree-index-all")
def read_root2():            
    query = characterist_image(directory_input + '\\uno1.JPG')
    data = read_characteristics(directory_dataset)
    k = 4
    radio = 5
    finds = search_rtree_indexed_all(query, data, radio )
    return finds


@app.get("/search-rtree-index-knn")
def read_root3():            
    query = characterist_image(directory_input + '\\uno1.JPG')
    data = read_characteristics(directory_dataset)
    k = 4
    radio = 5
    finds = search_rtree_indexed_knn(query, data, radio, k)
    return finds

# @app.get("/search-knn-hight")
# def read_root4():    
#     query = characterist_image(directory_input + '\\uno1.JPG')
#     data = read_characteristics(directory_dataset)
#     k = 4
#     componentes_principales = apply_pca(data, 10)
   
#     #finds = knn_secuencial_pq(query, data, k)
#     return componentes_principales 




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
