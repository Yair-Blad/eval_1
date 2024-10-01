import cv2
import matplotlib.pyplot as plt
import numpy as np

#todos las actividades del readme las cree en un solo archivo 

# Actividad 1
#primero leemos la imagen a escala de grises
img = cv2.imread('imagen_medica.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Imagen original: ')
plt.axis('off')
plt.show()

#se recorto la imagen a un tamaño de 100 x 100 px
recorte = img[0:100, 0:100]
resized = cv2.resize(img, (200, 200)) # redimensionamos la imagen a un tamaño de 200 x 200 px

#se imprime la imagen recortada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recorte, cmap='gray')
plt.title('Imagen recortada: ')
plt.axis('off')

# se imprime la imagen redimensionada
plt.subplot(1, 2, 2)
plt.imshow(resized, cmap='gray')
plt.title('Imagen redimensionada: ')
plt.axis('off')
plt.show()

#Actividad 2
# se aplican los filtros Gaussian Blur, sobel y canny
blurred = cv2.GaussianBlur(img, (5, 5), 0)
bordes = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
canny = cv2.Canny(img, 50, 150)

#se imprime la imagen original
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original: ')
plt.axis('off')

#imagen suavisada
plt.subplot(2, 2, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Imagen suavizada: ')
plt.axis('off')

# imagen con el filtro sobel
plt.subplot(2, 2, 3)
plt.imshow(bordes, cmap='gray')
plt.title('Imagen bordes sobel: ')
plt.axis('off')

# imagen con bordes canny
plt.subplot(2, 2, 4)
plt.imshow(canny, cmap='gray')
plt.title('Imagen bordes canny: ')
plt.axis('off')
plt.show()

#Actividad 3
# se genera el histograma de la imagen y se muestra
histograma_original = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title('Histograma original: ')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia')
plt.plot(histograma_original)
plt.xlim([0, 256])
plt.show()

#se genera la ecualización de la imagen
imagen_ecualizada = cv2.equalizeHist(img)
cv2.imshow('Imagen ecualizada: ', imagen_ecualizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# se aplica la ecualizacion a de histograma a la imagen
histograma_ecualizado = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])

#se muestra el histograma  ecualizado
plt.title('Histograma ecualizado: ')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia')
plt.plot(histograma_ecualizado)
plt.xlim([0, 256])
plt.show()


