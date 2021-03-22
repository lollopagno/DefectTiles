from Defect import crack_defect as crack
from Defect import pinhole_defct as pinhole
from Preprocessing import pre_processing as preprocess
import cv2 as cv

img = cv.imread("Resources/crackTile4.jpg")
cv.imshow("Img original", img)

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Preprocessing
img_pre_processing = preprocess.start(img_gray)
# Pinhole defect
pinhole.detect(img_pre_processing)
# Crack defect
# crack_detect = crack.detect(img_pre_processing)

cv.imshow("Img result canny", img_pre_processing)
# cv.imshow("Result Crack", crack_detect)
cv.waitKey(0)