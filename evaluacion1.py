import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen_medica.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Imagen original: ')
plt.axis('off')
plt.show()

recorte = img[0:100, 0:100]
resized = cv2.resize(img, (200, 200))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(recorte, cmap='gray')
plt.title('Imagen recortada: ')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(resized, cmap='gray')
plt.title('Imagen redimensionada: ')
plt.axis('off')
plt.show()

blurred = cv2.GaussianBlur(img, (5, 5), 0)
bordes = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
canny = cv2.Canny(img, 50, 150)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original: ')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Imagen suavizada: ')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bordes, cmap='gray')
plt.title('Imagen bordes sobel: ')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(canny, cmap='gray')
plt.title('Imagen bordes canny: ')
plt.axis('off')
plt.show()

histograma_original = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title('Histograma original: ')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia')
plt.plot(histograma_original)
plt.xlim([0, 256])
plt.show()

imagen_ecualizada = cv2.equalizeHist(img)
cv2.imshow('Imagen ecualizada: ', imagen_ecualizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

histograma_ecualizado = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])

plt.title('Histograma ecualizado: ')
plt.xlabel('Intensidad de pixeles')
plt.ylabel('Frecuencia')
plt.plot(histograma_ecualizado)
plt.xlim([0, 256])
plt.show()
