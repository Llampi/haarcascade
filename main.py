import cv2
import numpy as np
import imutils
import os

carpeta_positivos = 'p'
carpeta_negativos = 'n'

if not os.path.exists(carpeta_positivos):
    print('Carpeta creada: ', carpeta_positivos)
    os.makedirs(carpeta_positivos)

if not os.path.exists(carpeta_negativos):
    print('Carpeta creada: ', carpeta_negativos)
    os.makedirs(carpeta_negativos)

cap = cv2.VideoCapture(0, cv2.CAP_ANY)

x1, y1 = 190, 80
x2, y2 = 510, 400

count_positivos = len(os.listdir(carpeta_positivos))
count_negativos = len(os.listdir(carpeta_negativos))

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    imAux = frame.copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    objeto = imAux[y1:y2, x1:x2]
    objeto = imutils.resize(objeto, width=38)
    # print(objeto.shape)

    k = cv2.waitKey(1)
    if k == ord('p'):
        cv2.imwrite(carpeta_positivos+'/objeto_{}.jpg'.format(count_positivos), objeto)
        print('Imagen almacenada en carpeta positivos: ', 'objeto_{}.jpg'.format(count_positivos))
        count_positivos += 1
    if k == ord('n'):
        cv2.imwrite(carpeta_negativos+'/objeto_{}.jpg'.format(count_negativos), objeto)
        print('Imagen almacenada en carpeta negativos: ', 'objeto_{}.jpg'.format(count_negativos))
        count_negativos += 1
    if k == 27:
        break

    cv2.imshow('frame', frame)
    cv2.imshow('objeto', objeto)

cap.release()
cv2.destroyAllWindows()

