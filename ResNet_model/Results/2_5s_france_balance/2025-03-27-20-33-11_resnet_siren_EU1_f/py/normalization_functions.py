# -*- coding: utf-8 -*-
"""
Created by kruets at 12.12.2018 13:40

@author: kruets

Feature:
# compendium off different normalization functions
"""
import numpy as np
from scipy import stats
from ..logging_alf import getLogger

LOGGER = getLogger()


def normalizeFeaturesWithNormMat(features, normMat):
    """
    Normalize with mean (-) and std(/) matrices, for normalizing to mean zero and std of one with pre-defined values
    :param features: features will be nornmalized inplace
    :param normMat: mean mat at first and std mat at second index
    """
    features -= normMat[0]
    features /= normMat[1]


def normalizeFeaturesToZeroMean_perFeature(features, **normalize_parameter):
    """
    Normalizes feature matrix to zero mean per feature (pixel for 2D image)
    :param features: features will be nornmalized inplace
    :return normalization matrix with mean matrix in first and std matrix in second index
    """
    meanMat = np.mean(features, axis=0, keepdims=False)
    stdMat = np.std(features, axis=0, keepdims=False)

    stdMat[stdMat == 0] = 1
    features -= meanMat
    features /= stdMat

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat

def normalizeFeaturesToZeroMean(features, **normalize_parameter):
    """
    Normalizes feature matrix to zero mean for whole input (for example full 2D image).
    :param features: features will be nornmalized inplace
    :return normalization matrix with mean matrix in first and std matrix in second index
    """
    keep_dims = features.ndim > 2

    meanVal = np.mean(features, keepdims=keep_dims)
    stdVal = np.std(features, keepdims=keep_dims)
    features -= meanVal
    if stdVal == 0:
        stdVal = 1
    features /= stdVal

    meanMat = np.full(features.shape[1:], meanVal)
    stdMat = np.full(features.shape[1:], stdVal)

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat


def normalizeToMax1_perFeature(features, **normalize_parameter):
    """
    Normalize each feature to max 1 per feature / pixel for 2D input
    :param features: features will be nornmalized inplace
    :return: normMat: mean mat (zeros) at first and std mat (max values) at second index
    """
    keep_dims = features.ndim > 2

    stdMat = np.max(features, axis=0, keepdims=keep_dims)

    stdMat[stdMat == 0] = 1
    features /= stdMat
    meanMat = np.zeros(features.shape[1:])

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat

def normalizeToMax1(features, **normalize_parameter):
    """
    Normalize features to max 1 (globally)
    :param features: features will be nornmalized inplace
    :return: normMat: mean mat (zeros) at first and std mat (max values) at second index
    """
    # bring to min zero to cover full range and fix problems with negative numbers
    min_val = np.min(features)
    features -= min_val

    maxVal = np.max(features)
    if maxVal == 0:
        LOGGER.warning("Features seem to be all zero - please check")
        maxVal = 1
    features /= maxVal

    stdMat = np.full(features.shape[1:], maxVal)
    meanMat = np.full(features.shape[1:], min_val)

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat

def normalizeToMax1PerBin(features, **normalize_parameter):
    """ Normalizes maximum value of one per feature for 1D and frequency bin for 2D images (ignore time direction)
    Args:
        features(numpy array): feature matrix which will be normalized inplace!
    Returns:
        maxMat(int): matrix with max values for each bin.
    """
    # bring to min zero to cover full range and fix problems with negative numbers
    min_val = np.min(features)
    features -= min_val

    if features.ndim == 4:
        # first reshape to 3d-array --> all patches in one dimension
        examples, patches, bins, channels = features.shape
        reshapeMatrix = features.reshape(examples * patches, bins, channels)
        stdMat = np.max(reshapeMatrix, axis=0, keepdims=True)
        stdMat = np.repeat(stdMat, features.shape[1], axis=0)
    elif features.ndim == 2:
        stdMat = np.max(features, axis=0)
    else:
        raise NotImplementedError("normalizeToMax1PerBin: please implement for number your ndims")

    # avoid division by zero
    stdMat[stdMat == 0] = 1
    features /= stdMat
    meanMat = np.full(features.shape[1:], min_val)

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat

def normalizeToZeroMeanPerBin(features, **normalize_parameter):
    """ Normalizes features to zero mean and std of one for each frequency bin (ignore time direction).
    Args:
        features(numpy array): feature matrix which will be normalized inplace!
    Returns:
        maxMat(int): matrix with normalization values for each bin and example
        2D normMat.shape = (2,features.shape[1]), 4D normMat.shape = ([mean/std], [examples], [values], [channels])).
    """
    if features.ndim == 4:
        # first reshape to 3d-array --> all patches in one dimension
        examples, patches, bins, channels = features.shape
        reshapeMatrix = features.reshape(examples*patches, bins, channels)
        # calc mean and std for z-score over all bins
        meanMat = np.mean(reshapeMatrix, axis=0, keepdims=True)
        stdMat = np.std(reshapeMatrix, axis=0, keepdims=True)
        meanMat = np.repeat(meanMat, features.shape[1], axis=0)
        stdMat = np.repeat(stdMat, features.shape[1], axis=0)
    elif features.ndim == 2:
        meanMat = np.mean(features, axis=0)
        stdMat = np.std(features, axis=0)
    else:
        raise NotImplementedError("normalizeToMax1PerBin: please implement for number your ndims")

    # avoid division by zero
    stdMat[stdMat == 0] = 1
    features -= meanMat
    features /= stdMat

    meanMat = np.expand_dims(meanMat, axis=0)
    stdMat = np.expand_dims(stdMat, axis=0)
    normMat = np.vstack((meanMat, stdMat))
    return normMat


def normalizeToMaxWithMat(features, maxMat):
    """ Normalizes feature matrix by dividing with maximum value.

    Args:
        features(numpy array): feature matrix
        maxMat(int): maximum value
    """
    features /= maxMat

def normalizeWithZScorePerBin(features, **normalize_parameter):
    ''' Normalization with z-score function of scipy over special axis
    :param features:
        - 4d- o 2d ndim feature set
    :return:
        - z-scores/normalized features
    '''
    if features.ndim == 4:
        features = stats.zscore(features, axis=2)
    elif features.ndim == 2:
        features = stats.zscore(features, axis=0)
    else:
        raise NotImplementedError("normalizeToMax1PerBin: please implement for number your ndims")
    return features


def normalizeFeaturesUsingFile(features, **normalize_parameter):
    """
    Normalize with mean (-) and std(/) matrices using defined npy files.
    :param features: features will be nornmalized inplace
    :param normFile: file with normalization matrix
    """
    normFile = normalize_parameter["normFile"]
    if normFile.endswith(".npz"):
        full_feature_set = np.load(normFile)
        normMat = full_feature_set['normMat']
    elif normFile.endswith(".npy"):
        normMat = np.load(normFile)
    else:
        LOGGER.error("Unknown data type for normalization file")
        raise ValueError

    features -= normMat[0]
    features /= normMat[1]
    return normMat
