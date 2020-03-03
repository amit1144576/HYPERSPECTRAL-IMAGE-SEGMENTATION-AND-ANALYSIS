import numpy as np
from spectral import imshow, envi
import cv2 as cv
from spectral import imshow, view_cube, kmeans, principal_components, save_rgb, get_rgb

img = cv.imread('/Users/ramancheema/Downloads/WhatsApp Image 2020-03-02 at 3.42.01 PM.jpeg')
save_rgb('/Users/ramancheema/Documents/swi.jpg', img)
arr_img = img[:, :, 0:1]
arr_img/255
dim = arr_img.shape
nRows = dim[0]
nCol = dim[1]
nbands = dim[2]
count = 0
noRows_larRec=count
row_endPoint = 0
row_startPoint = 0
for i in range(0, nRows):
    for j in range(0, nCol):
        for k in range(0, nbands):
            x = arr_img[i, j, k]
            if x > 0:
                arr_img[i, :, :] = 1
for x in range(0, nRows):
 if arr_img[x, :, :].sum() == 320:
        count = count + 1
        if(noRows_larRec<num):
            noRows_larRec=num
            #larger white rectangle correspond to grain
            row_endPoint=x
 if(arr_img[x, :, :].sum() != 320):
     count=0
row_startPoint=row_endPoint-noRows_larRec

arr_img[0:row_startPoint,:,:]=0
arr_img[row_startPoint:row_endPoint,:,:]=1
arr_img[row_endPoint+1:nRows, :, :]=0

#getting middle grain by selecting columns
range=nCol/4
middle_section=range+(range+range)
arr_img[:,0:range,:]=0
arr_img[:,middle_section:nCol,:]=0

#mask
mask=np.ones((nRows,nCol))



for i in range(0, nRows):
    for j in range(0, nCol):
        for k in range(0, nbands):
            mask[i,j]=mask[i,j] *arr_img[i, j, k]
for i in range(0, nRows):
    for j in range(0, nCol):
        for k in range(0, 3):
            if(mask[i,j]==0):
                img[i,j,k]=img[i,j,k]*mask[i,j]
save_rgb('/Users/ramancheema/Documents/swir.jpg', img)
