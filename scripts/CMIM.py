# encoding=utf8
import sys
import numpy as np
import scipy as sp
import math


#def entropy(X):
#    """ Computes entropy of vector X. """
#    n = len(X)

#    if n <= 1:
#        return 0

#    counts = np.bincount(X)
#    probs = counts[np.nonzero(counts)] / float(n)
#    n_classes = len(probs)

#    if n_classes <= 1:
#        return 0

#    entropy = - np.sum(probs * np.log(probs)) / np.log(n_classes)

#    return entropy


def shan_entropy(c):
    """shannon entropy given the counts in the histogram (distribution)"""
    c_normalized = c/float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H

def calc_MI(X,Y,bins=100):
    """calculates mutual information between two vectors"""
    counts_XY = np.histogram2d(X,Y,bins)[0]
    counts_X = np.histogram(X,bins)[0]
    counts_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(counts_X)
    H_Y = shan_entropy(counts_Y)
    H_XY = shan_entropy(counts_XY)

    MI = H_X + H_Y - H_XY

    return MI


def calc_CMI(X,Y,Z,bins=100):
    """calculates conditional mutual information I(X,Y|Z) = H(X,Z)−H(Z)−H(X,Y,Z)+H(Y,Z)"""

    counts_XYZ = np.histogramdd([X,Y,Z], bins)[0]
    counts_XZ = np.histogram2d(X, Z, bins)[0]
    counts_YZ = np.histogram2d(Y, Z, bins)[0]
    counts_Z = np.histogram(Z, bins)[0]

    H_XYZ = shan_entropy(counts_XYZ)
    H_XZ = shan_entropy(counts_XZ)
    H_YZ = shan_entropy(counts_YZ)
    H_Z = shan_entropy(counts_Z)

    CMI = H_XZ - H_Z - H_XYZ + H_YZ

    return CMI


def fast_CMIM(target, data, number_of_features):
    """Performs the Conditional Mutual Information Maximization
        curMI //Called "ps"
  	count //Called "m"
  	chosen //Called "nu"
  	chosenMI //Track the winning MI scores
    """
    numFeatures = len(data.columns)
    numData = len(data)
    if number_of_features=='auto':
        number_of_features = numFeatures

    target = np.reshape(np.asarray(target),(1,numData))[0]

    chosenMI = [0]*numFeatures
    count = [0]*numFeatures
    CMIM = [0]*numFeatures
    curMI = [0]*numFeatures
    index = []

    for i in range(numFeatures):
        curMI[i] = calc_MI(np.reshape(np.asarray(data[[i]]),(1,numData))[0],target)
    
    for k in range(number_of_features):
        curBest= 0 #Called "s*, the best up do date score this iteration"  
        for n in range(numFeatures):
            while curMI[n] > curBest and count[n] < k:
                #count[n] = int(count[n]) + 1
                newIdx = int(CMIM[count[n]])
                newMI = calc_CMI(target, np.reshape(np.asarray(data[[n]]),(1,numData))[0],np.reshape(np.asarray(data[[newIdx]]),(1,numData))[0])
                if newMI < curMI[n]:
                    curMI[n] = newMI
                count[n] = int(count[n]) + 1
            if curMI[n] > curBest:
                curBest = curMI[n]
                CMIM[k] = n
                chosenMI[k] = curMI[CMIM[k]]

    for i in range(numFeatures):
        if CMIM[i] not in index:
            index.append(CMIM[i])
    for i in range(numFeatures):
        if i not in index:
            index.append(i)
                  
    return index, chosenMI


def slow_CMIM(target, data):
    """Performs the Conditional Mutual Information Maximization
        curMI //Called "ps"
        count //Called "m"
        CMIM //Called "nu"
        chosenMI //Track the winning MI scores
    """
    target = np.reshape(np.asarray(target),(1,numData))[0]

    numFeatures = len(data.columns)
    numData = len(data)

    chosenMI = [0]*numFeatures
    count = [0]*numFeatures
    CMIM = [0]*numFeatures
    curMI = [0]*numFeatures
    index = []

    for i in range(numFeatures):
        curMI[i] = calc_MI(np.reshape(np.asarray(data[[i]]),(1,numData))[0],target)

    for k in range(numFeatures):
        CMIM[k] = np.argsort(curMI)[::-1][0]
        chosenMI[k] = curMI[chosen[k]] 
        for n in range(numFeatures):
            newMI = calc_CMI(target, np.reshape(np.asarray(data[[n]]),(1,numData))[0],np.reshape(np.asarray(data[[CMIM[k]]]),(1,numData))[0])
            if newMI < curMI[n]:
                curMI[n] = newMI

    for i in range(numFeatures):
        if CMIM[i] not in index:
            index.append(CMIM[i])
    for i in range(numFeatures):
        if i not in index:
            index.append(i)

    return index, chosenMI   



def MIM(target, data):
    """Performs the Mutual Information Maximization"""

    numFeatures = len(data.columns)
    numData = len(data)

    target = np.reshape(np.asarray(target),(1,numData))[0]
    curMI = [0]*numFeatures

    for i in range(numFeatures):
        curMI[i] = calc_MI(np.reshape(np.asarray(data[[i]]),(1,numData))[0],target)

    MIM = np.argsort(curMI)[::-1]

    return MIM, curMI


def selectBestFeatures(target, data, method='CMIM', number_of_features='auto'):
    """select the best features given the target vector and data vector 
       method: CMIM or MI
       number_of_featuers: auto or integer. If 'auto' will automatically select features with some minimum information"""
    
    numFeatures = len(data.columns)
    numData = len(data)
 
    if method=='CMIM':
        index, scores = fast_CMIM(target, data, number_of_features)
    elif method=='MI':
        index, scores = MIM(target, data)

    if number_of_features=='auto':
        selected_indices = index[0:len([i for i in scores if i>0.0001])]
        selectedData = data[selected_indices]
        return selectedData
    else:
        selected_indices = index[0:min(int(number_of_features),numFeatures)]
        selectedData = data[selected_indices]
        return selectedData
        


