#!/usr/bin/env python

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from findtree import findtree
import time

# Image files to process
fname = ['nmzwj.png.jpg', 'aVZhC.png.jpg', '2K9EF.png.jpg',
         'YowlH.png.jpg', '2y4o5.png.jpg', 'FWhSP.png.jpg']
# fname = ['nmzwj.png.jpg']

# Initialize figures
fgsz = (16, 7)
figthresh = plt.figure(figsize=fgsz, facecolor='w')
figclust = plt.figure(figsize=fgsz, facecolor='w')
figcltwo = plt.figure(figsize=fgsz, facecolor='w')
figborder = plt.figure(figsize=fgsz, facecolor='w')
figthresh.canvas.set_window_title('Thresholded HSV and Monochrome Brightness')
figclust.canvas.set_window_title('DBSCAN Clusters (Raw Pixel Output)')
figcltwo.canvas.set_window_title('DBSCAN Clusters (Slightly Dilated for Display)')
figborder.canvas.set_window_title('Trees with Borders')

for ii, name in zip(range(len(fname)), fname):
    # Open the file and convert to rgb image
    rgbimg = np.asarray(Image.open(name))

    # Get the tree borders as well as a bunch of other intermediate values
    # that will be used to illustrate how the algorithm works
    t0 = time.time()
    borderseg, X, labels, Xslice = findtree(rgbimg)
    print(round(time.time() - t0, 3))

    # Display thresholded images
    axthresh = figthresh.add_subplot(2, 3, ii + 1)
    axthresh.set_xticks([])
    axthresh.set_yticks([])
    binimg = np.zeros((rgbimg.shape[0], rgbimg.shape[1]))
    for v, h in X:
        binimg[v, h] = 255
    axthresh.imshow(binimg, interpolation='nearest', cmap='Greys')

    # Display color-coded clusters
    axclust = figclust.add_subplot(2, 3, ii + 1)  # Raw version
    axclust.set_xticks([])
    axclust.set_yticks([])
    axcltwo = figcltwo.add_subplot(2, 3, ii + 1)  # Dilated slightly for display only
    axcltwo.set_xticks([])
    axcltwo.set_yticks([])
    axcltwo.imshow(binimg, interpolation='nearest', cmap='Greys')
    clustimg = np.ones(rgbimg.shape)
    unique_labels = set(labels)
    # Generate a unique color for each cluster
    plcol = cm.rainbow_r(np.linspace(0, 1, len(unique_labels)))
    for lbl, pix in zip(labels, Xslice):
        for col, unqlbl in zip(plcol, unique_labels):
            if lbl == unqlbl:
                # Cluster label of -1 indicates no cluster membership;
                # override default color with black
                if lbl == -1:
                    col = [0.0, 0.0, 0.0, 1.0]
                # Raw version
                for ij in range(3):
                    clustimg[pix[0], pix[1], ij] = col[ij]
                # Dilated just for display
                axcltwo.plot(pix[1], pix[0], 'o', markerfacecolor=col,
                             markersize=3, markeredgecolor=col)
    axclust.imshow(clustimg)
    axcltwo.set_xlim(0, binimg.shape[1] - 1)
    axcltwo.set_ylim(binimg.shape[0], -1)

    # Plot original images with read borders around the trees
    axborder = figborder.add_subplot(2, 3, ii + 1)
    axborder.set_axis_off()
    axborder.imshow(rgbimg, interpolation='nearest')
    for vseg, hseg in borderseg:
        axborder.plot(hseg, vseg, 'r-', lw=3)
    axborder.set_xlim(0, binimg.shape[1] - 1)
    axborder.set_ylim(binimg.shape[0], -1)

plt.show()
