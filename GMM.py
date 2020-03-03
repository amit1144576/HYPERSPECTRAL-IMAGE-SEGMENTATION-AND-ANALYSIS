import imageio as imageio
from scipy import misc, ndimage
from PIL import Image
import spectral.io.envi as envi
#j=imread('/Users/ramancheema/Downloads/orginal.jpg')
from spectral import imshow
img1 = envi.open('/Users/ramancheema/Downloads/038368_20000_us_2x_2014-11-14T124919_corr_rad.hdr','/Users/ramancheema/Downloads/038368_20000_us_2x_2014-11-14T124919_corr_rad.img')
j=imageio.imread('/Users/ramancheema/Downloads/orginal.jpg')
imshow(j)



