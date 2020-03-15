import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import pandas as pd

SWIRFileSource = "../../SegmentedImages/SourceFolder/SWIR.txt"
imageType = "SWIR"
imagesArray = []
with open(SWIRFileSource, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

for images in range(0, len(imagesArray)- len(imagesArray) + 2):
    imageIndex = images
    imageDirectory = "F:/SWIR/"
    imageName = imagesArray[imageIndex]

    # instantiating hyperspectral image object using spectral library
    SWIRimage = envi.open(imageDirectory + imageName + '.hdr',
                          imageDirectory + imageName + '.img')

    SWIRnumOfRows = SWIRimage.nrows
    SWIRnumOfCols = SWIRimage.ncols
    SWIRnumOfBands = SWIRimage.nbands

    imgSpectrum = int(str(SWIRnumOfBands))
    imgHeight = int(str(SWIRnumOfRows))
    imgWidth = int(str(SWIRnumOfCols))

    SWIRSegmentchannels = np.array([55, 41, 12])
    SWIRSegmentMax1 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[0])])
    SWIRSegmentMin1 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[0])])
    SWIRSegmentMax2 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[1])])
    SWIRSegmentMin2 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[1])])
    SWIRSegmentMax3 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[2])])
    SWIRSegmentMin3 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[2])])

    SWIRSegmentslice1 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    SWIRSegmentslice2 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    SWIRSegmentslice3 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmentslice1[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[0])] - SWIRSegmentMin1) /\
                                          (SWIRSegmentMax1 - SWIRSegmentMin1) * 255
                SWIRSegmentslice2[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[1])] - SWIRSegmentMin3) /\
                                          (SWIRSegmentMax3 - SWIRSegmentMin3) * 255
                SWIRSegmentslice3[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[2])] - SWIRSegmentMin2) /\
                                          (SWIRSegmentMax2 - SWIRSegmentMin2) * 255

    # ///////////////////////////////////////////////////// build RGB image ///////////////////////////////////////
    SWIRSegmentimage = np.zeros([SWIRnumOfRows, SWIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmentimage[r, c] = [SWIRSegmentslice1[r, c], SWIRSegmentslice2[r, c], SWIRSegmentslice3[r, c]]
    # cv2.imwrite("F:/SWIR/images/" + imageName + ".png", SWIRSegmentimage)

    # ///////////////////////////////////////////////////// segment grains ///////////////////////////////////////

    # nemo = cv2.imread("../../SegmentedImages/" + imageType + "/" + imageName + ".png")
    SWIRSegmentedrgb = SWIRSegmentimage
    # plt.imshow(segmentrgb)
    # plt.show()
    SWIRSegmented3d = cv2.cvtColor(SWIRSegmentedrgb, cv2.COLOR_BGR2RGB)
    # plt.imshow(segmented3d)
    # plt.show()
    SWIRSegmenthsvcoverted = cv2.cvtColor(SWIRSegmented3d, cv2.COLOR_RGB2HSV)
    SWIRSegment_lower_bounds = (0, 60, 50)
    SWIRSegment_upper_bounds = (100, 255, 255)
    SWIRSegmentmask = cv2.inRange(SWIRSegmenthsvcoverted, SWIRSegment_lower_bounds, SWIRSegment_upper_bounds)
    SWIRSegmentresult = cv2.bitwise_and(SWIRSegmented3d, SWIRSegmented3d, mask=SWIRSegmentmask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    SWIRSegmentresult = cv2.cvtColor(SWIRSegmentresult, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", result)

    SWIRSegmented = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmentresult[r, c, 0] or SWIRSegmentresult[r, c, 1] or SWIRSegmentresult[r, c, 2]:
                    SWIRSegmented[r, c] = SWIRSegmentresult[r, c, 2]

    # ////////////////////////////////////////// quadratic transformation SWIR //////////////////////////////////
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmented[r, c] > 0:
                    SWIRSegmented[r, c] = SWIRSegmented[r, c] * SWIRSegmented[r, c]

    SWIRSegmentedmaxValue = np.max(SWIRSegmented[:, :])
    SWIRSegmentedminValue = np.min(SWIRSegmented[:, :])

    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmented[r, c] = (SWIRSegmented[r, c] - SWIRSegmentedminValue) /\
                                      (SWIRSegmentedmaxValue - SWIRSegmentedminValue) * 250

    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmented[r, c] < 30:
                    SWIRSegmented[r, c] = 0

    # ///////////////////////////////////////////////// db-scan based algorithm //////////////////////////////////
    """
    applying DB-scan like algorithm for noise removal
    radius is based on Moore neighborhood
    """

    def db_scan_CleanGrains(radius, minPoints):
        labels = np.zeros((SWIRnumOfRows, SWIRnumOfCols))

        for r in range(0, SWIRnumOfRows):
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmented[r, c] > 0:
                    labels[r, c] = 1

        for r in range(radius - 1, SWIRnumOfRows - radius):
            for c in range(radius - 1, SWIRnumOfCols - radius):
                if SWIRSegmented[r, c] > 0:
                    numberOfNieghbors = 0
                    for radRow in range(r - radius + 1, r + radius):
                        for radCol in range(c - radius + 1, c + radius):
                            numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                    if numberOfNieghbors < minPoints:
                        SWIRSegmented[r, c] = 0


    # cleaning grains
    radius = 5
    minPoints = 50
    iterations = 1
    for i in range(0, iterations):
        db_scan_CleanGrains(radius, minPoints)

    # radius = 4
    # minPoints = 32
    # iterations = 3
    # for i in range(0, iterations):
    #     db_scan_CleanGrains(radius, minPoints)

    # ///////////////////////////////////////////// get middle grains of each spike ///////////////////////////////

    def selectpixels(midvertical, midhorizontal, radius, image):
        numberofpoints = 0
        minimumpoints = 2000
        expansionstep = 5
        SWIRSegmented = image
        for r in range(-radius, radius):
            for c in range(-radius, radius):
                if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                    numberofpoints = numberofpoints + 1
                    SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        for r in range(0, 25):
            if numberofpoints < minimumpoints and radius < 154:
                radius = radius + expansionstep
                numberofpoints = 0
                for r in range(-radius, radius):
                    for c in range(-radius, radius):
                        if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                            numberofpoints = numberofpoints + 1
                            SWIRSegmented[midvertical + r, midhorizontal + c] = 255

        # print(radius)
        # print(numberofpoints)

    selectpixels(525, 160, 30, SWIRSegmented)


    # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # export an excel file of the segmented data
    # np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', SWIRdataset, delimiter=',')
    # np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '1.csv', SWIRimage[:, :, 1], delimiter=',')

    print(str(images + 1) + " images of " + str(len(imagesArray)))