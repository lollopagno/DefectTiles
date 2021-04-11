from Defect import crack_defect as crack
from Defect import pinhole_defct as pinhole
from Defect import blob_defect as blob
from Preprocessing import pre_processing as preprocess
import cv2 as cv
import numpy as np

path_crack = "Resources/Crack"
path_blob = "Resources/BLob"
#img = cv.imread(f"{path_crack}/crackTile3.jpg")
img = cv.imread(f"{path_blob}/blobTile3.jpg")
cv.imshow("Img original", img)

method = "Canny"

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Preprocessing
img_pre_processing = preprocess.start(img_gray, edge_detection=method)

# Pinhole defect
# pinhole.detect(img_pre_processing)

# Blob defect
blob_detect = blob.detect(img, img=img_pre_processing, method=method)

# Crack defect
crack_detect = crack.detect(img_pre_processing / 255, method=method)

cv.imshow(f"Edge detection {method}", img_pre_processing)
cv.imshow("Result Blob", blob_detect)
cv.imshow("Result Crack", crack_detect)
cv.waitKey(0)
