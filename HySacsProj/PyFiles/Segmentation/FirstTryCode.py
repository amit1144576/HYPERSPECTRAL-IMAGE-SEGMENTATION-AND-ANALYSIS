import spectral.io.envi as envi
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

"""
Reading hyperspectral images data, exporting 2D-images and optionally excel file of the data
and Segmentation of grains
"""
# each image type SWIR or NIR can optionally as a package or one by one based on index number
# being executed and the result will be in appropriate folders. Note that later we need to add
# additional image names orderly in each text file.
SWIRFileSource = "../SegmentedImages/SourceFolder/SWIR.txt"
NIRFileSource = "../SegmentedImages/SourceFolder/NIR.txt"

# write 'SWIR' or 'NIR' for imageType
imageType = "SWIR"

if imageType == "SWIR":
    textFile = SWIRFileSource
elif imageType == "NIR":
    textFile = NIRFileSource

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
    firstSliceSpectrum = 0
    firstSliceMax = np.max(img[:, :, firstSliceSpectrum])
    firstSliceMin = np.min(img[:, :, firstSliceSpectrum])
    # last slice variants
    lastSliceSpectrum = imgSpectrum - 1
    lastSliceMax = np.max(img[:, :, lastSliceSpectrum])
    lastSliceMin = np.min(img[:, :, lastSliceSpectrum])
    # non-significant difference in spectrum need to be neglected
    # and make our processing time lower; spectrum values of grains
    # has huge difference in relation to the background surface
    differenceSpectrum = 0;
    # instantiating each slice and a segmented image
    firstSlice = np.zeros((imgRows, imgColumns))
    lastSlice = np.zeros((imgRows, imgColumns))
    segmented = np.zeros((imgRows, imgColumns))
    # comparision of both slice and filling segmented array
    sideBorder = 8

    for r in range(0, imgRows):
        # if 630 > r > 420:
        for c in range(0, imgColumns):
            # if imgColumns - sideBorder > c > sideBorder:
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
    # def thresholdImg():
    #
    #     segmented = cv2.imread("/Users/ahmad/PycharmProjects/HySacsProj/SegmentedImages/SWIR/038367.PNG", 0)
    #     thresholdvalue = 0
    #     maxvalue = 255
    #
    #     ret, thresh1 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     ret, thresh2 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     ret, thresh3 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    #     ret, thresh4 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    #     ret, thresh5 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    #
    #     titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    #     images = [segmented, thresh1, thresh2, thresh3, thresh4, thresh5]
    #
    #     for i in range(0, 6):
    #         plt.subplot(2, 3, i + 1)
    #         plt.imshow(images[i], cmap='gray')
    #         plt.title(titles[i])
    #         plt.xticks([])
    #         plt.yticks([])
    #     # plt.imshow(segmented)
    #     # plt.title(titles)
    #     plt.show()

    # thresholdImg()





    # def db_scan_CleanGrains(radius, minPoints):
    #     labels = np.copy(segmented)
    #
    #     for r in range(0, imgRows):
    #         for c in range(0, imgColumns):
    #             if segmented[r, c] > 0:
    #                 labels[r, c] = 1
    #
    #     for r in range(radius - 1, imgRows - radius):
    #         for c in range(radius - 1, imgColumns - radius):
    #             if 120 > segmented[r, c] > 0:
    #                 numberOfNieghbors = 0
    #                 for radRow in range(r - radius + 1, r + radius):
    #                     for radCol in range(c - radius + 1, c + radius):
    #                         numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
    #                 if numberOfNieghbors < minPoints:
    #                     segmented[r, c] = 0
    #
    #
    # # cleaning grains
    #
    # radius = 6
    # minPoints = 50
    # iterations = 3
    # for i in range(0, iterations):
    #     db_scan_CleanGrains(radius, minPoints)
    #
    # radius = 4
    # minPoints = 30
    # iterations = 10
    # for i in range(0, iterations):
    #     db_scan_CleanGrains(radius, minPoints)

    # export an excel file of the segmented data
    # np.savetxt("../../SegementedImages/" + imageType + "/" + imageName + '.csv', segmented, delimiter=',')

    """
    creating a 2d image using opencv library
    """
    cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", segmented)
    #
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
