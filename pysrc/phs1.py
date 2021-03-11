# This is the python source file to execute tasks performed in phase
# 1 of the project MD-TP3. It might be used in later phases as
# needed.

# Bibliotecas necessarias
import numpy as np
import os
import pandas as pd
import re
from skimage import io

# Get image file names
def getBenMalFiles():
    ben_dir = "data/train/benign/"
    mal_dir = "data/train/malignant/"
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

def getBenMalImgList():
    ben_dir, mal_dir, ben_files, mal_files = getBenMalFiles()
    
    ben_prefs, mal_prefs = getBenMalPrefixes(ben_files, mal_files)
    
    # Load images, this should take a while
    benImgList = []
    for i in range(len(ben_prefs)):
        benImgList.append(io.imread(ben_dir + ben_prefs[i] + ".jpg"))

    malImgList = []
    for i in range(len(mal_prefs)):
        malImgList.append(io.imread(mal_dir + mal_prefs[i] + ".jpg"))
        
    return benImgList, malImgList

# Does all the loading process
def loadImgDataset():
    benImgList, malImgList = getBenMalImgList()
    
    imgSize = 150528 # 224 x 224 pixels with 3 color dimensions each
    # The first column is supposed to be the class of the sample (0 for benign, 1 for malign)
    dataset = np.zeros((len(benImgList) + len(malImgList), 1 + imgSize), dtype=np.uint8)

    for row in range(len(benImgList)):
        dataset[row, 0] = 0 # The class benign
        dataset[row, 1:] = benImgList[row].flatten()
    for row in range(len(benImgList), dataset.shape[0]):
        dataset[row, 0] = 1 # The class malign
        dataset[row, 1:] = malImgList[row - len(benImgList)].flatten()
        
    return dataset

# Get singular values such that variance is preserved
#   up to 95%
def chooseBiggestSigmas(sigmas):
    chosenSigmas = list()
    head = 0
    while sum(chosenSigmas) / sigmas.sum() < 0.9999:
        chosenSigmas.append(sigmas[head])
        head += 1

    return np.array(chosenSigmas)

def buildReDf(df):
    u, sigmas, vh = np.linalg.svd(df, full_matrices=False)
    print("sigmas = ", sigmas)
    
    chosenSigmas = chooseBiggestSigmas(sigmas)
    print("chosenSigmas = ", chosenSigmas)
    
    dim = len(chosenSigmas) # Number of preserved dimensions
    aproxDf = u[::, :dim] @ np.diag(chosenSigmas)
    reDf = pd.DataFrame(aproxDf, columns=["pc" + str(i) for i in range(dim)])

    return reDf
