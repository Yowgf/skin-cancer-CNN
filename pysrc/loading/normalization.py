"""Routines for normalization of data"""

def zNormalize(df):
    sigma = df.std()
    mean = df.mean()

    return (df - mean) / sigma

def minMaxNormalize(df):
    mins = df.min()
    maxs = df.max()
    maxs[maxs == 0] = 1
    mins[mins == maxs] = 0

    df = (df - mins) / maxs
