"""Routines for dimensionality reduction"""

import numpy as np
import pandas as pd

# Dimensionality reduction based on svd
def chooseBiggestSigmas(sigmas, varRatio):
    chosenSigmas = list()
    head = 0
    
    # Goal is to get varRatio variance from original dataset
    while sum(chosenSigmas) / sigmas.sum() < varRatio:
        chosenSigmas.append(sigmas[head])
        head += 1

    return np.array(chosenSigmas)

def buildReDf(df):
    u, s, vh = np.linalg.svd(df, full_matrices=False)
    
    chosenSigmas = chooseBiggestSigmas(s, varRatio=0.95)    
    dim = len(chosenSigmas)
    
    reDf = pd.DataFrame(u[::, :dim] @ np.diag(s[:dim]), columns=["pc" + str(i) for i in range(dim)])

    return reDf
