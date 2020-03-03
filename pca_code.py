import math

import spectral.io.envi as envi
import numpy as np
import cv2
from skimage.color import rgb2gray
from spectral import imshow, view_cube, kmeans, principal_components, save_rgb, get_rgb
import cv2 as cv
img = envi.open('/Users/ramancheema/Documents/12cp/swir_0-25_374.jpg','/Users/ramancheema/Documents/12cp/VNIR/038369_20000_us_2x_2014-11-14T123117_corr_rad.img')
m=img.load()
#creating matrix
mm=m[:,:,0:25]

#incase of of specfic bands,like band no n to m-1, use  mm=m[:,:,n:m]

#this command create dimensions with eigen value
pc = principal_components(mm)

#oringal l=number of bands, l helps further in selecting number of band in output image
l=img.nbands

#covariance value
f=0.999
#showing image with covariance
#v = imshow(pc.cov)


#selecting appropriate value of covariance to get selected number of bands
while(l!=3):
 pc_fraction = pc.reduce(fraction=f)
 l=len(pc_fraction.eigenvalues) #how many egin vectors with selected co  variance
 if(l<3):
     f=f+0.00001
 if(l>3):
     f=f-0.00001

#reduce the dimensionality of the image pixels by projecting them onto the remaining eigenvectors.
img_pc = pc_fraction.transform(mm)

imshow(img_pc)

save_rgb('/Users/ramancheema/Documents/SWIR/rgb/swir_1D(0-25)_374.jpg', img_pc)

