#!/usr/bin/env python

from PIL import Image
import numpy as np
from findtree import findtree
import time
import cv2

# Image files to process
# fname = ['nmzwj.png.jpg', 'aVZhC.png.jpg', '2K9EF.png.jpg',
#          'YowlH.png.jpg', '2y4o5.png.jpg', 'FWhSP.png.jpg']
fname = ['nmzwj.png.jpg']

# Initialize figures
for ii, name in zip(range(len(fname)), fname):
    # Open the file and convert to rgb image
    rgbimg = np.asarray(Image.open(name))

    # Get the tree borders as well as a bunch of other intermediate values
    # that will be used to illustrate how the algorithm works
    border_seg, X, labels, Xslice = findtree(rgbimg)
    border = np.array(border_seg)[:, 0]
    print(border)
    # Display thresholded images
    binimg = np.zeros((rgbimg.shape[0], rgbimg.shape[1]), dtype=np.uint8)
    for v, h in X:
        binimg[v, h] = 255
    cv2.polylines(binimg, [border], True, 255)
    cv2.imshow('11', binimg)
    cv2.waitKey()

    # Display color-coded clusters
    clustimg = np.ones(rgbimg.shape)
    unique_labels = set(labels)
