import spectral.io.envi as envi
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

imageType = "SWIR"
imageName = "038375_SWIR_320m_SN3505_5000_us_2014-11-14T110520_corr_rad"
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

sliceNumber = 1
sliceMax = np.max(img[:, :, sliceNumber])
sliceMin = np.min(img[:, :, sliceNumber])
segmented = np.zeros((imgRows, imgColumns))
sideBorder = 8
for r in range(0, imgRows):
    if 630 > r > 420:
        for c in range(0, imgColumns):
            if imgColumns - sideBorder > c > sideBorder:
                segmented[r, c] = (img[r, c, sliceNumber] - sliceMin) / (sliceMax - sliceMin) * 255

# Make a quadratic transformation
for r in range(0, imgRows):
    for c in range(0, imgColumns):
        segmented[r, c] = segmented[r, c] * segmented[r, c]

maxValue = np.max(segmented[:, :])
minValue = np.min(segmented[:, :])

for r in range(0, imgRows):
    for c in range(0, imgColumns):
        segmented[r, c] = (segmented[r, c] - minValue) / (maxValue - minValue) * 255

# Second try to make quadratic transformation
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segmented[r, c] = segmented[r, c] * segmented[r, c]
#
# maxValue = np.max(segmented[:, :])
# minValue = np.min(segmented[:, :])
#
# for r in range(0, imgRows):
#     for c in range(0, imgColumns):
#         segmented[r, c] = (segmented[r, c] - minValue) / (maxValue - minValue) * 255

# zero padding
for r in range(0, imgRows):
    for c in range(0, imgColumns):
        if segmented[r, c] < 60:
            segmented[r, c] = 0
        # elif segmented[r, c] > 40:
        #     segmented[r, c] = 255


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# removing lines zero degrees
noOfPixels = 40
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            countPixels = 0
            for i in range(0, noOfPixels):
                if segmented[r, c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    segmented[r, c + d] = 0

# removing lines 275 degrees
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            countPixels = 0
            for i in range(0, noOfPixels):
                if segmented[r + i, c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    segmented[r + d, c + d] = 0

# removing lines 45 degrees
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            countPixels = 0
            for i in range(0, noOfPixels):
                if segmented[r - i, c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    segmented[r - d, c + d] = 0

# removing lines 2x/1y downward pattern
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            reverseCount = 0
            deleteCount = 0
            countPixels = 0
            for i in range(0, noOfPixels):
                reverseCount = reverseCount + 0.5
                if segmented[r + round(reverseCount), c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    deleteCount = deleteCount + 0.5
                    segmented[r + round(deleteCount), c + d] = 0

# removing lines 2x/1y upward pattern
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            reverseCount = 0
            deleteCount = 0
            countPixels = 0
            for i in range(0, noOfPixels):
                reverseCount = reverseCount + 0.5
                if segmented[r - round(reverseCount), c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    deleteCount = deleteCount + 0.5
                    segmented[r - round(deleteCount), c + d] = 0

# removing lines 10x/3y downward pattern
for r in range(0, imgRows):
    for c in range(0, imgColumns - noOfPixels):
        if segmented[r, c] > 0:
            reverseCount = 0
            deleteCount = 0
            countPixels = 0
            for i in range(0, noOfPixels):
                reverseCount = reverseCount + 0.3
                if segmented[r + round(reverseCount), c + i] > 0:
                    countPixels = countPixels + 1
            if countPixels == noOfPixels:
                for d in range(0, noOfPixels):
                    deleteCount = deleteCount + 0.3
                    segmented[r + round(deleteCount), c + d] = 0

# removing lines 10x/3y upward pattern
# for r in range(0, imgRows):
#     for c in range(0, imgColumns - noOfPixels):
#         if segmented[r, c] > 0:
#             reverseCount = 0
#             deleteCount = 0
#             countPixels = 0
#             for i in range(0, noOfPixels):
#                 reverseCount = reverseCount + 0.3
#                 if segmented[r - round(reverseCount), c + i] > 0:
#                     countPixels = countPixels + 1
#             if countPixels == noOfPixels:
#                 for d in range(0, noOfPixels):
#                     deleteCount = deleteCount + 0.3
#                     segmented[r - round(deleteCount), c + d] = 0

# //////////////////////////////////////////////Important: can remove as a square/////////////////////////
pixelLength = 20
for r in range(0, imgRows):
    for c in range(0, imgColumns - pixelLength):
        if  segmented[r, c] > 0:
            count = 0
            for l in range(0, pixelLength):
                for w in range(0, pixelLength):
                    if segmented[r + l, c + w] > 0:
                        count = count + 1
            if count == pixelLength * pixelLength:
                for d in range(0, pixelLength):
                    for e in range(0, pixelLength):
                        segmented[r + d, c + e] = 0

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# trying to remove the leaves
# density = 15
# for r in range(0, imgRows - density):
#     if 630 > r > 420:
#         for c in range(0, imgColumns - density):
#             count2 = 0
#             for x in range(0, density):
#                 if 130 > segmented[r + x, c + x] > 70:
#                     for y in range(0, density):
#                         count2 = count2 + 1  # segmented[r + x, c + x]
#
#             if count2 == density * density:
#                 for y in range(0, density):
#                     segmented[r + y, c + y] = 0

# ///////////////////////////////////////////////////////////////////////////////////////////////////////
"""
applying DB-scan algorithm for segmentation purpose radius is based on Moore neighborhood
"""
def db_scan_CleanGrains(radius, minPoints):
    labels = np.copy(segmented)

    for r in range(0, imgRows):
        for c in range(0, imgColumns):
            if segmented[r, c] > 0:
                labels[r, c] = 1

    for r in range(radius - 1, imgRows - radius):
        for c in range(radius - 1, imgColumns - radius):
            if segmented[r, c] > 0:
                numberOfNieghbors = 0
                for radRow in range(r - radius + 1, r + radius):
                    for radCol in range(c - radius + 1, c + radius):
                        numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                if numberOfNieghbors < minPoints:
                    segmented[r, c] = 0

# cleaning grains
radius = 4
minPoints = 20
iterations = 1
for i in range(0, iterations):
    db_scan_CleanGrains(radius, minPoints)

radius = 20
minPoints = 30
iterations = 1
for i in range(0, iterations):
    db_scan_CleanGrains(radius, minPoints)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# export the image as png file
cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", segmented)

# export an excel file of the segmented data
# np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', segmented, delimiter=',')
