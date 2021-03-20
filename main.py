from Defect import crack_defect as crack
import pre_processing as preprocess
import cv2 as cv

img = cv.imread("Resources/crackTile3.jpg")
cv.imshow("Img original", img)

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Pre-processing
img_pre_processing = preprocess.start(img_gray)
# Crack defect
crack_detect = crack.detect(img_pre_processing)

cv.imshow("Img result canny", img_pre_processing)
cv.imshow("Result Crack", crack_detect)
cv.waitKey(0)

# https://stackoverflow.com/questions/26332883/how-to-find-all-connected-components-in-a-binary-image-in-matlab
# https://stackoverflow.com/questions/52153979/feature-extraction-and-take-color-histogram
