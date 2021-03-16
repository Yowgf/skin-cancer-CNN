"""Some printing and plotting functions"""

from pysrc import loading

from skimage import io
import matplotlib.pyplot as plt

# Shows information on how the benign / malign classes are separated
#   in the data.
def showSeparation(classes):
    numBen = (classes == 0).sum() # Number of benign cases
    numMal = (classes == 1).sum() # Number of malign cases
    numTot = len(classes)
    
    print("Number of benign cases: ", numBen, " ({:.2f}%)".format(100 * numBen / numTot))
    print("Number of malignant cases: ", numMal, " ({:.2f}%)".format(100 * numMal / numTot))

def showOriginalImagesBenMal(testTrain, saveDir=None):
    ben_dir, mal_dir, ben_files, mal_files = loading.getBenMalFiles(testTrain)
    
    path_ben = [ben_dir + ben_file for ben_file in ben_files]
    path_mal = [mal_dir + mal_file for mal_file in mal_files]
    
    imgBen = io.imread(path_ben[0])
    imgMal = io.imread(path_mal[0])
    
    fig = plt.figure(figsize=(10, 10))
    
    # Show benign case
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Benign")
    plt.imshow(imgBen)
    
    # Show malign case
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Malign")
    plt.imshow(imgMal)
    
    if saveDir != None:
        plt.savefig(saveDir + "/ben-mal-diff.png")
