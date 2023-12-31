import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live

"# Detección de objetos"
"### Mantén la cámara fija en el objeto a detectar"

image = camera_input_live()

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Cargar el archivo haarcascade.xml
    cascade_path = "cascade.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Detectar objetos usando el clasificador de cascada
    objects = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(250, 250))

    if len(objects) > 0:
        st.write("# Objeto detectado")
        
        for (x, y, w, h) in objects:
            # Dibujar cuadro alrededor del objeto
            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        with st.expander("Show details"):
            st.write("Object(s) detected:", len(objects))

        st.image(cv2_img)
    else:
        st.write("Aún no se encuentra ningún objeto.")
