# # Convertir la imagen a escala de grises
# gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

# # Suavizado de la imagen
# gris = cv2.GaussianBlur(gris, (21, 21), 0)

# # Resta absoluta
# resta = cv2.absdiff(fondo, gris)

# # Aplicamos el umbral a la imagen
# umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)

# # Buscamos contorno en la imagen
# im, contornos, hierarchy = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# Importamos las librerías necesarias
import numpy as np
import cv2
import time
import os
import skimage.io as io

# Cargamos el vídeo Yini Yfin Xini Xfin
pos_cam = [(.3 , .5, .43,.53)]
videos = ["offi_lima2.MP4"]

# Inicializamos el primer frame a vacío.
# Nos servirá para obtener el fondo
fondo = None

# Recorremos todos los frames

save_path = os.path.join(os.getcwd(),"img_full3")
n_img = 1436
count = 0 
#print(os.getcwd())
for video, pos in zip(videos,pos_cam):
	camara = cv2.VideoCapture(os.path.join("img_orig",video))
	while True:
		# Obtenemos el frame
		(grabbed, frame) = camara.read()
		if not grabbed:
			break

		(alto, ancho, chs) = frame.shape
		#print("ancho",ancho)
		#print("alto", alto) .25,.5,.6,.7
		y_ini = int(alto*pos[0])
		y_fin = int(alto*pos[1])
		x_ini = int(ancho*pos[2])
		x_fin = int(ancho*pos[3])
		frame = frame[y_ini:y_fin,x_ini:x_fin]
		# # Si hemos llegado al final del vídeo salimos
		
		if count == 2:
			n_img +=1
			#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imwrite(os.path.join(save_path,"%d.png"%n_img),frame)
			count=0
		count +=1

		#time.sleep(0.0001)
		# cv2.imshow("Camara", frame)
		# key = cv2.waitKey(1) & 0xFF

		# if key == ord("s"):
		#  	break

# Liberamos la cámara y cerramos todas las ventanas
camara.release()
cv2.destroyAllWindows()