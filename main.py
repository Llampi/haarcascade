import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live

"# Streamlit camera input live Demo"
"## Try holding a qr code in front of your webcam"

image = camera_input_live()

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    detector = cv2.QRCodeDetector()

    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.write("# Found QR code")
        st.write(data)

        # Dibujar cuadro alrededor del QR code
        if bbox is not None:
            bbox = np.int0(bbox)  # Convertir a tipo de datos entero
            cv2.polylines(cv2_img, [bbox], isClosed=True, color=(255, 0, 0), thickness=2)

        with st.expander("Show details"):
            st.write("BBox:", bbox)
            st.write("Straight QR code:", straight_qrcode)

        st.image(cv2_img)
    else:
        st.write("No QR code found in the image.")

