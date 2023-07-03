import streamlit as st
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Para imágenes estáticas:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # gray
img_file_buffer = st.camera_input("Take a picture")

import math


# 
def calculate_angle(point_a, point_b, point_c):
    # Obtener las coordenadas (x, y) de los puntos
    ax, ay = point_a.x, point_a.y
    bx, by = point_b.x, point_b.y
    cx, cy = point_c.x, point_c.y

    # Calcular los vectores entre los puntos
    vector_ab = (bx - ax, by - ay)
    vector_bc = (cx - bx, cy - by)

    # Calcular los productos internos de los vectores
    dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]

    # Calcular las magnitudes de los vectores
    magnitude_ab = math.sqrt(vector_ab[0] ** 2 + vector_ab[1] ** 2)
    magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)

    # Calcular el coseno del ángulo utilizando el producto interno y las magnitudes
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)

    # Calcular el ángulo en radianes
    angle_rad = math.acos(cos_angle)

    # Convertir el ángulo a grados
    angle_deg = math.degrees(angle_rad)

    return angle_deg

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape

        # Convertir la imagen BGR a RGB antes de procesar.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        # Detectar las coordenadas de la nariz.
        nose_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height

        print(f'Nose coordinates: ({nose_x}, {nose_y})')

        annotated_image = image.copy()

        # Dibujar la segmentación en la imagen.
        # Para mejorar la segmentación alrededor de los bordes, considera aplicar un filtro bilateral conjunto
        # a "results.segmentation_mask" con "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # Dibujar los landmarks del pose en la imagen.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        # Graficar los landmarks del pose en el mundo.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# Para la entrada de la cámara web:
# ...

# Para la entrada de la cámara web:
#cap = cv2.VideoCapture(0)
#cap = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
cap = st.image(img_file_buffer)
contador = 0
sentadilla = False
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # Si estás cargando un video, utiliza 'break' en lugar de 'continue'.
            continue

        # Para mejorar el rendimiento, opcionalmente marca la imagen como no editable para pasarla por referencia.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Dibujar la anotación del pose en la imagen.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Detectar sentadillas
        if results.pose_landmarks is not None:
            # Obtener las coordenadas de los landmarks relevantes
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calcular los ángulos entre los landmarks relevantes
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Verificar si se está realizando una sentadilla
            if left_knee_angle < 90 and right_knee_angle < 90:
                
                if (sentadilla  == False):
                    contador += 1
                    sentadilla = True
                
            else:
                sentadilla = False

        # Voltear la imagen horizontalmente para mostrarla en modo selfie.
        text = "Sentadilla detectada: " + str(contador)
        letras = []
        #for letra in text:
        #    letras.append(letra[::-1])
        #flipped_text = text[::-1]  # Invertir el texto manualmente
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

