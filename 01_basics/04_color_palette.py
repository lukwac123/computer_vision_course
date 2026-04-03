import cv2
import numpy as np


def nothing(x):
    pass

img = np.zeros(shape=(300, 500, 3), dtype='uint8')
cv2.namedWindow('Paleta')

# Utworzenie pasków przewijania
cv2.createTrackbar('Red', 'Paleta', 0, 255, nothing)
cv2.createTrackbar('Green', 'Paleta', 0, 255, nothing)
cv2.createTrackbar('Blue', 'Paleta', 0, 255, nothing)

while True:
    cv2.imshow(winname='Paleta', mat=img)

    # pobierz aktualną pozyscję
    r = cv2.getTrackbarPos('Red', 'Paleta')
    g = cv2.getTrackbarPos('Green', 'Paleta')
    b = cv2.getTrackbarPos('Blue', 'Paleta')

    img[:] = [b, g, r]

    # Dodaj nazwę i wartość kolorów na obrazie
    cv2.putText(img, f'Red: {r}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(img, f'Green: {g}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f'Blue: {b}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow('Paleta', img)
    if cv2.waitKey(20) == 27:
        break
