import numpy as np
from spectral import imshow, envi
import cv2 as cv
from spectral import imshow, view_cube, kmeans, principal_components, save_rgb, get_rgb

img1 = cv.imread('/Users/ramancheema/Downloads/WhatsApp Image 2020-02-23 at 9.58.24 AM.jpeg')
img=cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
arr_img=cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
#arr_img = img[:, :, 0:1]

dim = arr_img.shape
nRows = dim[0]
nCol = dim[1]
#nbands = dim[2]
num = 0
n=num
e = 0
s = 0
diff = 0

#selecting biggest white area
for i in range(0, nRows):
    for j in range(0, nCol):
        #for k in range(0, nbands):
            x = arr_img[i, j]
            if x > 0:
                arr_img[i, :] = 1
for x in range(0, nRows):
 if arr_img[x, :].sum() == 320:
        num = num + 1
        if(n<num):
            n=num
            e=x
 if(arr_img[x, :].sum() != 320):
     num=0
s=e-n


#selection of biggest white area
arr_img[0:s,:]=0
arr_img[s:e,:]=1
arr_img[e+1:nRows, :]=0
#imshow(arr_img)

#preparing mask
arr_img[:,0:80]=0
arr_img[:,241:320]=0
mask=np.ones((800,320))
e1=80
s1=241
#multiplying mask with image
for i in range(0, nRows):
    for j in range(0, nCol):
        #for k in range(0, nbands):
            mask[i,j]=mask[i,j] *arr_img[i, j]
for i in range(0, nRows):
    for j in range(0, nCol):
        #for k in range(0, 3):
            if(arr_img[i,j]==0):
                img[i,j]=img[i,j]*arr_img[i,j]

#imshow(img)
s1=80
e1=240
#grid
#grid
#sizeX=4
#sizeY=4
#for i in range(0,nRows):
    #for j in range(0, nCol):
        #roi = img[i*nRows:i*sizeY/nRows + sizeY/nRows ,j*sizeX/mCols:j*sizeX/mCols + sizeX/mCols]
if(e-s<100):
    e=e+(100-(e-s))
if(e1-s1<180):
    e1=e1+(180-(e1-s1))
a=img[s:e,s1:e1]
imshow(a)
save_rgb('/Users/ramancheema/Documents/grid_img2.jpg', a)
