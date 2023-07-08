import uvicorn
from fastapi import FastAPI
import nest_asyncio
import os
import time
from src.search_knn_secuencial import knn_secuencial_rango, knn_secuencial_pq ,return_images
from src.face_reconition import characterist_image, read_characteristics, convert_base64TOImage
from src.search_R_tree_index import search_rtree_indexed_all, search_rtree_indexed_knn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware (
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

nest_asyncio.apply()


current_dir = os.path.dirname(os.path.abspath(__file__))
directory_dataset = os.path.join(current_dir, "src", "index")
directory_input = os.path.join(current_dir, "src", "input")


# print(f"Main: Ruta actual = {current_dir}" )
# print(f"Main: directory_dataset = {directory_dataset}" )
# print(f"Main: directory_input = {directory_input}" )

@app.post("/search-secuencial-pq")
def read_root(request_body: dict):
    start_time = time.time()  
    
    try:
        imagen_base64 = request_body.get("imagen")
        cantidad = request_body.get("cantidad")
        
        if imagen_base64 is None or cantidad is None:
            raise ValueError("Parámetros 'imagen' y 'cantidad' requeridos en el cuerpo de la solicitud.")
        
        query = convert_base64TOImage(imagen_base64)
        if query is None:
            raise ValueError("No se pudo convertir la imagen a base64.")
        
        data = read_characteristics(directory_dataset)
        k = cantidad
        finds = knn_secuencial_pq(query, data, k)
        execution_time = time.time() - start_time  

        result = {
            "success": True,
            "execution_time": execution_time,
            "data": return_images(finds)
        }
        return result
    except ValueError as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result
    except Exception as e:
        result = {
            "success": False,
            "error_message": "Error en el servidor"
        }
        return result



@app.post("/search-secuencial-range")
def read_root1(request_body: dict):
    try:
        imagen_base64 = request_body.get("imagen")
        radio = request_body.get("radio")
        cantidad = request_body.get("cantidad")
        
        if imagen_base64 is None or radio is None or cantidad is None:
            raise ValueError("Parámetros 'imagen', 'radio' y 'cantidad' requeridos en el cuerpo de la solicitud.")
        
        query = convert_base64TOImage(imagen_base64)
        if query is None:
            raise ValueError("No se pudo convertir la imagen a base64.")
        
        data = read_characteristics(directory_dataset)
        k = cantidad
        finds = knn_secuencial_rango(query, data, radio, k)
        
        result = {
            "success": True,
            "data": return_images(finds)
        }
        return result
    except ValueError as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result
    except Exception as e:
        result = {
            "success": False,
            "error_message": "Error en el servidor"
        }
        return result


@app.post("/search-rtree-index-all")
def read_root2(request_body: dict):
    try:
        imagen_base64 = request_body.get("imagen")
        radio = request_body.get("radio")
        cantidad = request_body.get("cantidad")
        
        if imagen_base64 is None or radio is None or cantidad is None:
            raise ValueError("Parámetros 'imagen', 'radio' y 'cantidad' requeridos en el cuerpo de la solicitud.")
        
        query = convert_base64TOImage(imagen_base64)
        if query is None:
            raise ValueError("No se pudo convertir la imagen a base64.")
        
        data = read_characteristics(directory_dataset)
        k = cantidad
        finds = search_rtree_indexed_all(query, data, radio)
        
        result = {
            "success": True,
            "data": finds
        }
        return result
    except ValueError as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result
    except Exception as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result



@app.post("/search-rtree-index-knn")
def read_root3(request_body: dict):
    try:
        imagen_base64 = request_body.get("imagen")
        radio = request_body.get("radio")
        cantidad = request_body.get("cantidad")
        
        if imagen_base64 is None or radio is None or cantidad is None:
            raise ValueError("Parámetros 'imagen', 'radio' y 'cantidad' requeridos en el cuerpo de la solicitud.")
        
        query = convert_base64TOImage(imagen_base64)
        if query is None:
            raise ValueError("No se pudo convertir la imagen a base64.")
        
        data = read_characteristics(directory_dataset)
        k = cantidad
        finds = search_rtree_indexed_knn(query, data, radio, k)
        
        result = {
            "success": True,
            "data": finds
        }
        return result
    except ValueError as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result
    except Exception as e:
        result = {
            "success": False,
            "error_message": str(e)
        }
        return result


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
