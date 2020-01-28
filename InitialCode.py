import spectral as spectral
import spectral.io.envi as envi
from matplotlib.pyplot import imshow
# *from spectral import *

#import cv2
import numpy as np
img = envi.open('C:/Users/Ram/Desktop/DataSet/swir/038369_SWIR_320m_SN3505_5000_us_2014-11-14T123117_corr_rad.hdr','C:/Users/Ram/Desktop/DataSet/swir/038369_SWIR_320m_SN3505_5000_us_2014-11-14T123117_corr_rad.img')
print(img)

print(img.shape)
for i in range(0,5) :
    for j in range(0,2) :
        pixel=img[i,j] #i, j are spatial coordinates
        print(pixel.shape)
        print('The value at'+str(i)+str(j)+ 'is:',pixel)

#pixel=img[799,319]
#print(pixel)
#print(pixel.shape)
#print(pixel.size)
#imshow(float(img))


#We are stuck here. We get the values located at each pixel insde the datacube. What has to be done with these values?
