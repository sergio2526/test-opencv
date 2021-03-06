from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points(pts):
    # ordenar los puntos según sus coordenadas x
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # agarre los puntos más a la izquierda y más a la derecha del orden
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    #ahora, ordena las coordenadas más a la izquierda según su
    # Coordenadas y para que podamos tomar la parte superior izquierda y la inferior izquierda
    # puntos, respectivamente
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # ahora que tenemos la coordenada superior izquierda, utilícela como
    # ancla para calcular la distancia euclidiana entre el
    # puntos superior izquierdo y derecho; por el pitagórico
    # teorema, el punto con la mayor distancia será
    # nuestro punto inferior derecho
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # devuelve las coordenadas en la parte superior izquierda, superior derecha,
    # orden de abajo a la derecha y de abajo a la izquierda
    return np.array([tl, tr, br, bl], dtype = "float32")

def four_point_transform(image, pts):
    # obtener un orden coherente de los puntos y descomprimirlos
    # individualmente
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # calcular el ancho de la nueva imagen, que será el
    # distancia máxima entre la parte inferior derecha y la parte inferior izquierda
    # coordenadas x o las coordenadas x superior derecha y superior izquierda
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # calcular la altura de la nueva imagen, que será la
    # distancia máxima entre la parte superior derecha y la inferior derecha
    # coordenadas y o las coordenadas y superior izquierda e inferior izquierda
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # ahora que tenemos las dimensiones de la nueva imagen, construya
    # el conjunto de puntos de destino para obtener una "vista de pájaro",
    # (es decir, vista de arriba hacia abajo) de la imagen, nuevamente especificando puntos
    # en la parte superior izquierda, superior derecha, inferior derecha e inferior izquierda
    # orden
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calcula la matriz de transformación de perspectiva y luego aplícala
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # devuelve la imagen deformada
    return warped