import spectral.io.envi as envi
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

"""
Reading hyperspectral NIR type images data, exporting 2D-images and optionally excel file of the data
and Segmentation of grains
"""
# this class is for later use to make the code object oriented
class VNIRSegmentation:
    def __init__(self, name, sourcefile):
        self.name = name
        self.sourcefile = sourcefile


VNIRFileSource = "../SegmentedImages/SourceFolder/NIR.txt"
imageType = "NIR"
textFile = VNIRFileSource

imagesArray = []
with open(textFile, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

for images in range(0, len(imagesArray)):

    imageIndex = images
    imageDirectory = "/Users/ahmad/Downloads/HySacs/" + imageType + "/"
    imageName = imagesArray[imageIndex]

    # instantiating hyperspectral image object using spectral library
    img = envi.open(imageDirectory + imageName + '.hdr',
                    imageDirectory + imageName + '.img')

    # reading and instantiating variables of number of rows, columns and bands of the image
    imgRows = img.nrows
    imgColumns = img.ncols
    imgBands = img.nbands

    imgSpectrum = int(str(imgBands))

    """
    making a comparision between first and last slices of spectrum and clearing points 
    which has not significant difference
    """
    # first slice variants
    firstSliceSpectrum = imgSpectrum - 1
    firstSliceMax = np.max(img[:, :, firstSliceSpectrum])
    firstSliceMin = np.min(img[:, :, firstSliceSpectrum])
    # last slice variants
    lastSliceSpectrum = 0
    lastSliceMax = np.max(img[:, :, lastSliceSpectrum])
    lastSliceMin = np.min(img[:, :, lastSliceSpectrum])
    # non-significant difference in spectrum need to be neglected
    # and make our processing time lower; spectrum values of grains
    # has huge difference in relation to the background surface
    differenceSpectrum = 60;
    # instantiating each slice and a segmented image
    firstSlice = np.zeros((imgRows, imgColumns))
    lastSlice = np.zeros((imgRows, imgColumns))
    segmented = np.zeros((imgRows, imgColumns))
    # comparision of both slice and filling segmented array

    for r in range(0, imgRows):
        for c in range(0, imgColumns):
            if r < 1190:
                firstSlice[r, c] = (img[r, c, firstSliceSpectrum] - firstSliceMin) / \
                                   (firstSliceMax - firstSliceMin) * 256
                lastSlice[r, c] = (img[r, c, lastSliceSpectrum] - lastSliceMin) / \
                                  (lastSliceMax - lastSliceMin) * 256
                if firstSlice[r, c] > lastSlice[r, c] + differenceSpectrum:
                    segmented[r, c] = (img[r, c, firstSliceSpectrum] - firstSliceMin) / \
                                      (firstSliceMax - firstSliceMin) * 256
                if firstSlice[r, c] < lastSlice[r, c] - differenceSpectrum:
                    segmented[r, c] = (img[r, c, firstSliceSpectrum] - firstSliceMin) / \
                                      (firstSliceMax - firstSliceMin) * 256

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

    """
    applying DB-scan algorithm for segmentation purpose
    radius is based on Moore neighborhood
    """

    def db_scan_CleanGrains(radius, minPoints):
        labels = np.copy(segmented)

        for r in range(0, imgRows):
            for c in range(0, imgColumns):
                if segmented[r, c] > 0:
                    labels[r, c] = 1

        for r in range(radius - 1, imgRows - radius):
            for c in range(radius - 1, imgColumns - radius):
                if 120 > segmented[r, c] > 0:
                    numberOfNieghbors = 0
                    for radRow in range(r - radius + 1, r + radius):
                        for radCol in range(c - radius + 1, c + radius):
                            numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                    if numberOfNieghbors < minPoints:
                        segmented[r, c] = 0


    # cleaning grains

    radius = 6
    minPoints = 50
    iterations = 3
    for i in range(0, iterations):
        db_scan_CleanGrains(radius, minPoints)

    radius = 4
    minPoints = 30
    iterations = 10
    for i in range(0, iterations):
        db_scan_CleanGrains(radius, minPoints)

    # export an excel file of the segmented data
    # np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', segmented, delimiter=',')

    """
    creating a 2d image using opencv library
    """
    cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", segmented)

    print(str(images + 1) + " images of " + str(len(imagesArray)))

# plt.plot(segmented)
# plt.xlabel("rows")
# plt.ylabel("columns")
# plt.show()

# time = [x for x in range(len(segmented[615, :]))]
# plt.xlabel("rows")
# plt.ylabel("columns")
# plt.bar(segmented[615, :], time)
# plt.show()
#
# plt.scatter(segmented[600, :], segmented[615, :])
# plt.show()
