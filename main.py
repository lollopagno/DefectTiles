import crack_defect as crack
import pre_processing as preprocess
import cv2 as cv

img = cv.imread("Resources/crackTile.jpg")
# cv.imshow("Img original", img)

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

img_pre_processing = preprocess.start(img_gray)
# TODO aggiungere conteggio dei pixel neri per essere confrontato con l'immagine di test
a = [[1,     1,     0,     0,     0,     0,     0,
     1,     1,     0,     0,     1,     1,     0,
     1,     1,     0,     0,     0,     1,     0,
     1,     1,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     1,     0,
     0,     0,     0,     0,     0,     0,     0]]

crack_detect = crack.detect(img_pre_processing)


cv.imshow("Img result canny", img_pre_processing)
cv.waitKey(0)

# https://stackoverflow.com/questions/26332883/how-to-find-all-connected-components-in-a-binary-image-in-matlab
