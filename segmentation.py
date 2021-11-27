import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from sklearn.cluster import  estimate_bandwidth
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import tiffile

def normalize(img):
    for i in range(0,len(img[0][0])):
        max = np.amax(img[:,:,i])
        for x in range(len(img)):
            for y in range(len(img[x])):
                img[x][y][i]=int((img[x][y][i]/max)*256)
    return img

def transform(img):
    X = len(img)
    Y = len(img[0])
    c = len(img[0][0])
    timg = np.zeros([X*Y,c])
    ind=0
    for x in range(0,X):
        for y in range(0,Y):
            for i in range(0,c):
                timg[ind][i] = img[x][y][i]
            
    return timg


#path of the input image
path = "Dataset/AnnualCrop/AnnualCrop_5.tif"
img = tiffile.imread(path)
img = np.dstack([img[:,:,2],img[:,:,3],img[:,:,7]])
img = normalize(img)

timg = transform(img)
bw = estimate_bandwidth(timg,quantile=0.7)
msc = MeanShift(bandwidth=bw, bin_seeding=True)
msc.fit(timg)
print("number of estimated clusters : %d" % len(np.unique(msc.labels_)))


labels = msc.labels_
result_image = np.reshape(labels, [len(img),len(img[1])])
fig = plt.figure(2, figsize=(14, 12))
ax = fig.add_subplot(121)
ax = plt.imshow(img) 
ax = fig.add_subplot(122)
ax = plt.imshow(result_image)  
plt.show()