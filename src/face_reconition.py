import face_recognition as fr
import os
from sklearn.decomposition import PCA
import base64
from PIL import Image
import io
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))


dataset_dev = os.path.join(current_dir, "dataset", "images_dev")
dataset_local = os.path.join(current_dir, "dataset", "images_local1")

output= "index"
# print(f"Face_recognition: Ruta actual = {current_dir}" )
# print(f"Face_recognition: dataset_dev = {dataset_dev}" )
# print(f"Face_recognition: dataset_local = {dataset_local}" )

def characterist_image(imagen_path):
    imagen = fr.load_image_file(imagen_path)
    caracteristicas = fr.face_encodings(imagen)
    if len(caracteristicas) > 0: # Es rostro
        return caracteristicas[0]    
    # No hay rostros
    return None

def convert_base64TOImage(imagen_base64):
    imagen_decodificada = base64.b64decode(imagen_base64)
    imagen_pil = Image.open(io.BytesIO(imagen_decodificada))
    imagen_np = np.array(imagen_pil)
    caracteristicas = fr.face_encodings(imagen_np)
    if len(caracteristicas) > 0: # Es rostro
        return caracteristicas[0]    
    # No hay rostros
    return None


def characterist_all_images_dir(dataset):
    vector_characterist = {}
    for name in os.listdir(dataset):
        directorio_persona = os.path.join(dataset, name)
        if os.path.isdir(directorio_persona):
            caracteristicas_persona = []

            for imagen_nombre in os.listdir(directorio_persona):
                imagen_path = os.path.join(directorio_persona, imagen_nombre)
                caracteristicas = characterist_image(imagen_path)
                if caracteristicas is not None:
                    caracteristicas_persona.append(caracteristicas.tolist())
            if caracteristicas_persona:
                vector_characterist[name] = caracteristicas_persona

    return vector_characterist

def save_characteristics(caracteristicas, archivo_salida):
    with open(archivo_salida, "w") as file:
        for indice, (nombre_persona, caracteristicas_persona) in enumerate(caracteristicas.items(), start=1):
            nombre_persona_indice = f"{nombre_persona}_{str(indice).zfill(4)}"
            for caracteristica in caracteristicas_persona:
                file.write(nombre_persona_indice + ", " + ", ".join(str(x) for x in caracteristica) + "\n")

    print("Vectores característicos guardados en el archivo:", archivo_salida)

def read_characteristics(file_path):
    vector_characterist = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(", ")
                nombre_persona_indice = parts[0]
                nombre_persona, indice = nombre_persona_indice.rsplit("_", 1)
                indice = int(indice.lstrip("0"))
                caracteristicas = [float(x) for x in parts[1:]]

                if nombre_persona not in vector_characterist:
                    vector_characterist[nombre_persona] = []

                vector_characterist[nombre_persona].append(caracteristicas)
    ##print(vector_characterist)
    return vector_characterist





#dataset_local = characterist_all_images_dir(dataset_local)
#save_characteristics(dataset_local, output )
#read_characteristics("index")
#print(dataset_local)
