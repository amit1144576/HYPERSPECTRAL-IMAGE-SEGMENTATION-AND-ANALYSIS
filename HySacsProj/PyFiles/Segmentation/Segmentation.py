import numpy as np

"""
Reading hyperspectral SWIR type images data, exporting 2D-images and optionally excel file of the data
and Segmentation of grains
"""
# this class is for later use to make the code object oriented; two images of types VNIR and SWIR
# are going to be given as input
class Segmentation:

    def __init__(self, vnirimagename, vnirsource, swirimagename, swirsource):
        self.swirimagename = swirimagename
        self.vnirsource = vnirsource
        self.vnirimagename = vnirimagename
        self.swirsource = swirsource


