import spectral.io.envi as envi
import numpy as np
from spectral import imshow
import cv2 as cv
from spectral import imshow, view_cube, kmeans, principal_components, save_rgb, get_rgb
img_SWIR = envi.open('/Users/ramancheema/Documents/12cp/SWIR/038368_SWIR_320m_SN3505_5000_us_2014-11-14T124919_corr_rad.hdr','/Users/ramancheema/Documents/12cp/SWIR/038368_SWIR_320m_SN3505_5000_us_2014-11-14T124919_corr_rad.img')
img_VNIR=envi.open('/Users/ramancheema/Documents/12cp/VNIR/038368_20000_us_2x_2014-11-14T124919_corr_rad.hdr','/Users/ramancheema/Documents/12cp/VNIR/038368_20000_us_2x_2014-11-14T124919_corr_rad.img')
s=img_SWIR[:,:,0:160]
v=img_VNIR[:,:,0:256]
width=s.shape[1] #storing original number of column
height=s.shape[0] #storing original number of rows
#width= number of columns
#height=number of rows

scale_percent_cols=100
scale_percent_rows=100
#to get number of col=1600 since they are 320
while(width!=1600):
 scale_percent_cols += 1
 width = int(s.shape[1] * scale_percent_cols / 100)

#to get number of row=3200 since they are 800
while(height!=3200):
 scale_percent_rows+=1
 height = int(s.shape[0] * scale_percent_rows/ 100)

dim = (width, height)
# resize image
#img = np.array([[ 86  83 101 142]
             #   [162 103 144 151]
              #  [125 154 189  67]
               # [138 116 124  43]], dtype=np.uint8)
#enlarged = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
#print(enlarged)
# Result:
#[[ 86  86  83  83 101 101 142 142]
# [ 86  86  83  83 101 101 142 142]
# [162 162 103 103 144 144 151 151]
# [162 162 103 103 144 144 151 151]
# [125 125 154 154 189 189  67  67]
# [125 125 154 154 189 189  67  67]
# [138 138 116 116 124 124  43  43]
# [138 138 116 116 124 124  43  43]]



#INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, and INTER_LANCZOS4.
resized = cv.resize(s, dim, interpolation=cv.INTER_AREA)

print('Resized Dimensions : ', resized.shape)
save_rgb('/Users/ramancheema/Documents/resized.jpg', resized)

#save_rgb('/Users/ramancheema/Desktop/INTER_AREA.jpg', resized)



