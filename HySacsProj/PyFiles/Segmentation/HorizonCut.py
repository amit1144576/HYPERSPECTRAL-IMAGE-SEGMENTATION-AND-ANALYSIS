import spectral.io.envi as envi
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

imageType = "SWIR"
imageName = "038370_SWIR_320m_SN3505_5000_us_2014-11-14T120124_corr_rad"
imageDirectory = "/Users/ahmad/Downloads/HySacs/" + imageType + "/"
img = envi.open(imageDirectory + imageName + '.hdr',
                imageDirectory + imageName + '.img')

# reading and instantiating variables of number of rows, columns and bands of the image
imgRows = img.nrows
imgColumns = img.ncols
imgBands = img.nbands

imgSpectrum = int(str(imgBands))
imgHeight = int(str(imgRows))
imgWidth = int(str(imgColumns))

sliceNumber = 505
sliceMax = np.max(img[sliceNumber, :, :])
sliceMin = np.min(img[sliceNumber, :, :])
segmented = np.zeros((imgColumns, imgBands))

for c in range(0, imgColumns):
    for b in range(0, imgBands):
        segmented[c, b] = (img[sliceNumber, c, b] - sliceMin) / (sliceMax - sliceMin) * 250


cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", segmented)



