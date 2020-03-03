import spectral.io.envi as envi
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

imageType = "SWIR"
imageName = "038367_SWIR_320m_SN3505_5000_us_2014-11-14T122812_corr_rad"
imageDirectory = "/Users/ramkishore/Downloads/HySacs/" + imageType + "/"
img = envi.open(imageDirectory + imageName + '.hdr',
                imageDirectory + imageName + '.img')

# reading and instantiating variables of number of rows, columns and bands of the image
imgRows = img.nrows
imgColumns = img.ncols
imgBands = img.nbands

imgSpectrum = int(str(imgBands))
imgHeight = int(str(imgRows))
imgWidth = int(str(imgColumns))

slice1Number = 1
slice1Max = np.max(img[:, :, slice1Number])
slice1Min = np.min(img[:, :, slice1Number])

slice50Number = 50
slice50Max = np.max(img[:, :, slice50Number])
slice50Min = np.min(img[:, :, slice50Number])

slice100Number = 20
slice100Max = np.max(img[:, :, slice100Number])
slice100Min = np.min(img[:, :, slice100Number])

slice150Number = 150
slice150Max = np.max(img[:, :, slice150Number])
slice150Min = np.min(img[:, :, slice150Number])

slice200Number = 200
slice200Max = np.max(img[:, :, slice200Number])
slice200Min = np.min(img[:, :, slice200Number])

slice250Number = 250
slice250Max = np.max(img[:, :, slice250Number])
slice250Min = np.min(img[:, :, slice250Number])

segSlice1 = np.zeros((imgRows, imgColumns))
segSlice50 = np.zeros((imgRows, imgColumns))
segSlice100 = np.zeros((imgRows, imgColumns))
segSlice150 = np.zeros((imgRows, imgColumns))
segSlice200 = np.zeros((imgRows, imgColumns))
segSlice250 = np.zeros((imgRows, imgColumns))
sideBorder = 10
for r in range(0, imgRows):
    # if 630 > r > 420:
        for c in range(0, imgColumns):
            # if imgColumns - sideBorder > c > sideBorder:
                segSlice1[r, c] = (img[r, c, slice1Number] - slice1Min) / (slice1Max - slice1Min) * 255
                segSlice50[r, c] = (img[r, c, slice50Number] - slice50Min) / (slice50Max - slice50Min) * 255
                segSlice100[r, c] = (img[r, c, slice100Number] - slice100Min) / (slice100Max - slice100Min) * 255
                segSlice150[r, c] = (img[r, c, slice150Number] - slice150Min) / (slice150Max - slice150Min) * 255
                segSlice200[r, c] = (img[r, c, slice200Number] - slice200Min) / (slice200Max - slice200Min) * 255
                segSlice250[r, c] = (img[r, c, slice250Number] - slice250Min) / (slice250Max - slice250Min) * 255



# # Make a quadratic transformation
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segSlice1[r, c] = segSlice1[r, c] * segSlice1[r, c]
#
# maxValue = np.max(segSlice1[:, :])
# minValue = np.min(segSlice1[:, :])
#
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segSlice1[r, c] = (segSlice1[r, c] - minValue) / (maxValue - minValue) * 255
#
# # Second try to make quadratic transformation
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segSlice1[r, c] = segSlice1[r, c] * segSlice1[r, c]
#
# maxValue = np.max(segSlice1[:, :])
# minValue = np.min(segSlice1[:, :])
#
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segSlice1[r, c] = (segSlice1[r, c] - minValue) / (maxValue - minValue) * 255
#
# # zero padding
# threshold = 50
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         if segSlice1[r, c] < threshold:
#             segSlice1[r, c] = 0
#         # elif segmented[r, c] > 40:
#         #     segmented[r, c] = 255
#
# # //////////////////////////////////////////////Important: can remove as a square/////////////////////////
# pixelLength = 25
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - pixelLength):
#         if  segSlice1[r, c] > 0:
#             count = 0
#             for l in range(0, pixelLength):
#                 for w in range(0, pixelLength):
#                     if segSlice1[r + l, c + w] > 0:
#                         count = count + 1
#             if count == pixelLength * pixelLength:
#                 for d in range(0, pixelLength):
#                     for e in range(0, pixelLength):
#                         segSlice1[r + d, c + e] = 0
#
# # //////////////////////////////////////////////// Remove solid lines /////////////////////////////////////
# # removing lines zero degrees
# noOfPixels = 50
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 if segSlice1[r, c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     segSlice1[r, c + d] = 0
#
# # removing lines 275 degrees
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 if segSlice1[r + i, c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     segSlice1[r + d, c + d] = 0
#
# # removing lines 45 degrees
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 if segSlice1[r - i, c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     segSlice1[r - d, c + d] = 0
#
# # removing lines 2x/1y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.5
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.5
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 2x/1y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.5
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.5
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/3y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.3
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.3
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/3y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.3
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.3
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/2y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.2
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.2
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/2y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.2
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.2
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/1y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.1
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.1
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/1y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.1
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.1
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/9y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.9
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.9
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/9y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.9
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.9
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/8y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.8
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.8
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/8y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.8
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.8
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/7y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.7
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.7
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/7y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.7
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.7
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/6y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.6
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.6
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/6y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.6
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.6
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # removing lines 10x/4y downward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.4
#                 if segSlice1[r + round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.4
#                     segSlice1[r + round(deleteCount), c + d] = 0
#
# # removing lines 10x/4y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segSlice1[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.4
#                 if segSlice1[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.4
#                     segSlice1[r - round(deleteCount), c + d] = 0
#
# # ///////////////////////////////////////////////////////////////////////////////////////////////////////
# """
# applying DB-scan algorithm for segmentation purpose radius is based on Moore neighborhood
# """
# def db_scan_CleanGrains(radius, minPoints):
#     labels = np.copy(segSlice1)
#
#     for r in range(0, imgRows):
#         for c in range(0, imgColumns):
#             if segSlice1[r, c] > 0:
#                 labels[r, c] = 1
#
#     for r in range(radius - 1, imgRows - radius):
#         for c in range(radius - 1, imgColumns - radius):
#             if segSlice1[r, c] > 0:
#                 numberOfNieghbors = 0
#                 for radRow in range(r - radius + 1, r + radius):
#                     for radCol in range(c - radius + 1, c + radius):
#                         numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
#                 if numberOfNieghbors < minPoints:
#                     segSlice1[r, c] = 0
#
# # cleaning grains
# radius = 4
# minPoints = 20
# iterations = 1
# for i in range(0, iterations):
#     db_scan_CleanGrains(radius, minPoints)
#
# radius = 20
# minPoints = 30
# iterations = 1
# for i in range(0, iterations):
#     db_scan_CleanGrains(radius, minPoints)
#
# # ///////////////////////////////////////////////////////  EXPORT  ////////////////////////////////////////
# # export the image as png file
# cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", segSlice100)
# export an excel file of the segmented data
# np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', segmented, delimiter=',')

# /////////////////////////////////////////////////////export as RGB ///////////////////////////////////////
image = np.zeros([imgRows, imgColumns, 3], dtype=np.uint8)
for r in range(0, imgRows):
    # if 630 > r > 420:
        for c in range(0, imgColumns):
            image[r, c] = [segSlice1[r, c], segSlice50[r, c], segSlice100[r, c]]
# cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", image)

# /////////////////////////////////////////////////Train data using PCA ////////////////////////////////////
# image = StandardScaler().fit_transform(image)

newImage = image.reshape(imgRows * imgColumns, 3)
# cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", newImage)

# note: no need to scale data since our data doesnt have different scales
# scaledData = preprocessing.scale(newImage.T) # one type of scaling
# reduced = StandardScaler().fit_transform(newImage) # second type of scaling should use "fit_transform(newImage.T)"
reduced = PCA(n_components=2).fit_transform(newImage.T)
print(newImage.shape)

# print(reduced)

# np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', reduced, delimiter=',')
