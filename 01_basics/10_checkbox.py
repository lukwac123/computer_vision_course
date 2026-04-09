import cv2
import numpy as np


img = cv2.imread(r'01_basics/assets/checkbox.png')
cv2.imshow('img', img)
cv2.waitKey(0)

img = cv2.copyMakeBorder(
    src=img,
    top=20,
    bottom=20,
    left=20,
    right=20,
    borderType=cv2.BORDER_CONSTANT,
    value=(255, 255, 255)
)

cv2.imshow('img', img)
cv2.waitKey(0)

grey = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', grey)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(src=grey, ksize=(5, 5), sigmaX=0)
cv2.imshow('blurred', blurred)
cv2.waitKey(0)

thresh = cv2.threshold(src=blurred, thresh=75, maxval=200, type=cv2.THRESH_BINARY)[1]
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

contours = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
print(f'[INFO] Liczba wszystkich konturów: {len(contours)}')

img_cnt = cv2.drawContours(image=img.copy(), contours=[contours[1]], contourIdx=-1, color=(0, 255, 0), thickness=2)
cv2.imshow('img_cnt', img_cnt)
cv2.waitKey(0)

# wyszukanie konturu z zaznaczonym checkboxem
checked_idx = None
total = 0

for idx in [1, 2]:
    # wygenerowanie maski
    mask = np.zeros(shape=grey.shape, dtype='uint8')
    cv2.drawContours(mask, [contours[idx]], contourIdx=-1, color=255, thickness=-1)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask_inv', mask_inv)
    cv2.waitKey(0)

    answer = cv2.add(grey, mask_inv)
    cv2.imshow('answer', answer)
    cv2.waitKey(0)

    ansver_inv = cv2.bitwise_not(src=answer)
    cv2.imshow('answer_inv', ansver_inv)
    cv2.waitKey(0)

    cnt = cv2.countNonZero(ansver_inv)
    if cnt > total:
        checked_idx = idx

    print(checked_idx)

img = cv2.drawContours(image=img, contours=[contours[checked_idx]], contourIdx=-1, color=(0, 255, 0), thickness=2)
cv2.imshow('img', img)
cv2.waitKey(0)