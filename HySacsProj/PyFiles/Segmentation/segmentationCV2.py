import spectral.io.envi as envi
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================================================
# NOTE!!!!! this page was used just to try opencv for segmentation but our own build code is working better
# =========================================================================================================
SWIRFileSource = "../SegmentedImages/SourceFolder/SWIR.txt"
imageType = "SWIR"
textFile = SWIRFileSource

imagesArray = []
with open(textFile, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())
for images in range(0, len(imagesArray)):
    imageIndex = images
    imageDirectory = "/Users/ahmad/Downloads/HySacs/" + imageType + "/"
    imageName = imagesArray[imageIndex]

    img = envi.open(imageDirectory + imageName + '.hdr',
                        imageDirectory + imageName + '.img')
    imgRows = img.nrows
    imgColumns = img.ncols
    imgBands = img.nbands
    imgSpectrum = int(str(imgBands))

    firstSliceSpectrum = 0
    firstSliceMax = np.max(img[:, :, firstSliceSpectrum])
    firstSliceMin = np.min(img[:, :, firstSliceSpectrum])
    # last slice variants
    lastSliceSpectrum = imgSpectrum - 1
    lastSliceMax = np.max(img[:, :, lastSliceSpectrum])
    lastSliceMin = np.min(img[:, :, lastSliceSpectrum])

    differenceSpectrum = 20;

    imgSpectrum = int(str(imgBands))
    firstSlice = np.zeros((imgRows, imgColumns))
    lastSlice = np.zeros((imgRows, imgColumns))
    segmented = np.zeros((imgRows, imgColumns))

    for r in range(0, imgRows):
        for c in range(0, imgColumns):
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

    cv2.imwrite("../SegmentedImages/" + imageType + "/" + imageName + ".png", segmented)
    print(str(images + 1) + " images of " + str(len(imagesArray)))

    def thresholdImg():
        segmented = cv2.imread("../SegmentedImages/" + imageType + "/" + imageName + ".png", 0)
        thresholdvalue = 0
        maxvalue = 255

        ret, thresh1 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh2 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, thresh3 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        ret, thresh4 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        ret, thresh5 = cv2.threshold(segmented, thresholdvalue, maxvalue, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)

        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [segmented, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(0, 6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()

    # thresholdImg()

    def gaussian():
        segmented = cv2.imread("../SegmentedImages/" + imageType + "/" + imageName + ".png", 0)
        segmented = cv2.medianBlur(segmented, 5)

        ret, th1 = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(segmented, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(segmented, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [segmented, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    # gaussian()

    def complex():
        segmented = cv2.imread("../SegmentedImages/" + imageType + "/" + imageName + ".png", 0)

        # global thresholding
        ret1, th1 = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2, th2 = cv2.threshold(segmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(segmented, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [segmented, 0, th1,
                  segmented, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
                  'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
                  'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

        for i in range(3):
            plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
            plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
            plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
            plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
        plt.show()

    complex()

