import numpy as np
import cv2

def translate(image, x, y):
	# Definir la matriz de traducción y realizar la traducción
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Devuelve la imagen traducida
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Toma las dimensiones de la imagen
	(h, w) = image.shape[:2]

	# Si el centro es Ninguno, inicialícelo como el centro de
	# la imagen
	if center is None:
		center = (w / 2, h / 2)

	# Realiza la rotación
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Devuelve la imagen rotada
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# inicializar las dimensiones de la imagen que se cambiará de tamaño y
	# toma el tamaño de la imagen
	dim = None
	(h, w) = image.shape[:2]

	# si tanto el ancho como el alto son Ninguno, devuelva el
	# imagen original
	if width is None and height is None:
		return image

	# compruebe si el ancho es Ninguno
	if width is None:
	# calcular la relación de la altura y construir el
	# dimensiones
		r = height / float(h)
		dim = (int(w * r), height)

	# de lo contrario, la altura es Ninguna
	else:
	# calcular la relación del ancho y construir el
	# dimensiones
		r = width / float(w)
		dim = (width, int(h * r))

	# cambiar el tamaño de la imagen
	resized = cv2.resize(image, dim, interpolation = inter)

	# devuelve la imagen redimensionada
	return resized