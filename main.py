from Defect import crack_defect as crack
from Defect import pinhole_defct as pinhole
from Defect import blob_defect as blob
from Preprocessing import pre_processing as preprocess
import cv2 as cv
import numpy as np
import sys

path = "Resources/Blob"
img = cv.imread(f"{path}/blobTile.jpg")
cv.imshow("Img original", img)

method = "Sobel"

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Preprocessing
img_pre_processing = preprocess.start(img_gray, method=method)

# Pinhole defect
# pinhole.detect(img_pre_processing)

# Blob defect
blob_detect = blob.detect(img_pre_processing, size_blob=300, method=method)

# Crack defect
#crack_detect = crack.detect(img_pre_processing, method=method)

#cv.imshow(f"Edge detection {method}", img_pre_processing)
#cv.imshow("Result Blob", blob_detect)
cv.waitKey(0)
