"""Takes care of all of data loading. These functions were primarily
built during phase 1, but should be used in latter phases as
needed"""

import numpy as np
import os
import pandas as pd
import re
from skimage import io

# Get image file names
def getBenMalFiles(trainTest):
    if trainTest not in {"train", "test"}:
        raise AttributeError("Invalid argument ***" + trainTest +
                             "*** only options are 'train' and 'test'")
    
    ben_dir = "data/" + trainTest + "/benign/"
    mal_dir = "data/" + trainTest + "/malignant/"
    ben_files = np.sort(os.listdir(ben_dir))
    mal_files = np.sort(os.listdir(mal_dir))
    
    return ben_dir, mal_dir, ben_files, mal_files

# Fetch image files prefixes
def sortByNumPrefix(fileNames):
    
    prefixes = []
    for i in range(len(fileNames)):
        prefixes.append(re.sub("\..*", "", fileNames[i]))
    
    return np.sort(np.array(prefixes).astype(int)).astype(str)

def getBenMalPrefixes(ben_files, mal_files):
    ben_prefs = sortByNumPrefix(ben_files)
    mal_prefs = sortByNumPrefix(mal_files)
    
    return ben_prefs, mal_prefs

def getBenMalImgList(trainTest):
    ben_dir, mal_dir, ben_files, mal_files = getBenMalFiles(trainTest)
    
    ben_prefs, mal_prefs = getBenMalPrefixes(ben_files, mal_files)
    
    # Load images, this should take a while
    benImgList = []
    for i in range(len(ben_prefs)):
        benImgList.append(io.imread(ben_dir+ben_prefs[i]+".jpg"))

    malImgList = []
    for i in range(len(mal_prefs)):
        malImgList.append(io.imread(mal_dir+mal_prefs[i]+".jpg"))
        
    return benImgList, malImgList

# Receives nimages x (npixels x npixels x colorDim) flattened matrix
def sampleImages(imgsMatrix):
    print("Sampling images...")
    origVar = imgsMatrix.var()
    print("Original variance: {:.2f}".format(origVar))
    
    numImgs, nX, nY, nD = imgsMatrix.shape
    
    b1, b2 = (4, 4) # Size of the sampling window
    sampImgs = np.zeros((numImgs, nX // b1, nY // b2, nD), dtype=np.uint8)
    
    # Must iterate through the rows, b1 by b1
    for row in np.arange(nX, step=b1):
        for col in np.arange(nY, step=b2):
            sampImgs[:, row//b1, col//b2] = imgsMatrix[:, row:row+b1, col:col+b2].mean(axis=1).mean(axis=1)

    print("Finished sampling.")
    sampVar = sampImgs.var()
    print("Final variance: {:.2f}".format(sampVar))
    print("Loss: {:.2f}%".format(100 * (1 - sampVar / origVar)))
    
    return sampImgs

def prod(arr):
    maxSize = 100
    arrLen = len(arr)
    if len(arr) > maxSize:
        raise AttributeError("prod() is only to be used for small 1D" +
                             " arrays. Array received is " + str(arrLen) +
                             " long.")
    acc = 1
    for i in range(len(arr)):
        acc *= arr[i]
    
    return acc

# Does all the loading process
# Returns the column of labels of the dataset, along with the
# block-sampled images.
def loadImgDataset(trainTest):
    benImgList, malImgList = getBenMalImgList(trainTest)

    benSize = len(benImgList)
    malSize = len(malImgList)

    numImgs = benSize + malSize
    imgDims = (224, 224, 3)
    dataset = np.zeros(np.append(numImgs, imgDims), dtype=np.uint8)
    dataset[:benSize] = np.array(benImgList)
    dataset[benSize:] = np.array(malImgList)

    classes = np.array(np.append(np.repeat(0, benSize), np.repeat(1, malSize)))
    sampData = sampleImages(dataset)
    
    # Flatten the data into a 2D array.
    sampDims = sampData.shape[1:]
    flatData = np.zeros((numImgs, prod(sampDims)), dtype=np.uint8)
    for i in range(numImgs):
        flatData[i] = sampData[i].flatten()
    
    return classes, flatData
