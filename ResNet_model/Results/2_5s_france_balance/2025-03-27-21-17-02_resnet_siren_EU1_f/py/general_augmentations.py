# general_augmentations is a module to store general augmentations functions
# for use within data generators.
#
# Additional Required Packages for random rotate, electric transform, and
# grid distortion:
#   conda install -c conda-forge albumentations

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..core.helper import helper
from .augmentation_utils import *

from albumentations import GridDistortion, Rotate, ElasticTransform

__author__ = 'David Johnson'
__copyright__ = "Fraunhofer IDMT"


### Custom Augmentations ###
def mixup(x, y, **kwargs):
    """
    Performs mixup between two images from x (numpy array of images/spectrograms)
    and y (numpy array of labels, one hot encoded).
    From: https://arxiv.org/abs/1710.09412

    Args:
        x: numpy array of images or spectrograms with len >= 2
        y: numpy array of one hot encoded labels with len >= 2
        alpha: float value for beta distribution

    returns:
        x: numpy arrays of mixed up images (No original data is included)
        y: numpy arrays of mixed up labels (No original data is included)
    """

    aug = mixup_augmentor(**kwargs)
    return aug(x, y)


def mixup_multilabel(x, y, **kwargs):
    """
    Performs mixup between two images from x (numpy array of images/spectrograms).
    Targets y are combined using logical OR operation (hence they are not scaled
    by alpha factor). The main idea is that in scenarios such as polyphonic sound
    event detection (SED), we still want to detect sound events, even if their
    loudness is strongly decreased during the mixup process)

    Args:
        x: numpy array of images or spectrograms with len >= 2
        y: numpy array of one hot encoded labels with len >= 2
        alpha: float value for beta distribution

    returns:
        x: numpy arrays of mixed up images (No original data is included)
        y: numpy arrays of mixed up labels (No original data is included)
    """

    aug = mixup_multilabel_augmentor(**kwargs)
    return aug(x, y)


def random_erasing(x, y, **kwargs):
    """
    Randomly selects rectangular regions from the image/spectrogram to erase and replace
    pixels with random values (i.e. noise).
    From: https://arxiv.org/abs/1708.04896

    Args:
        x: image
        y: label
        p: probablity of erasing being performed
        s_l: min proportion of erased area to input image
        s_h: max proportion of erased area to input image
        r_1: minimum aspect ratio of erased area
        r_2: maximum aspect ratio of erased area
        v_l: minimum value for erased area
        v_h: maximum value for erased area
        pixel_level: random values to fill in erased image at the pixel level, or patch level

    returns:
        x: transformed features
        y: labels
    """

    aug = get_random_eraser(v_l=np.min(x), v_h=np.max(x), **kwargs)
    for i in range(len(x)):
        x[i] = aug(x[i])
    return x, y


def spec_augment(x, y, **kwargs):
    """
    Performs SpecAugment without time warping. Time wrapping omitted since it is
    very expensive and author claims it doesn't make much of a difference.
        Paper: https://arxiv.org/abs/1904.08779
        Code:  https://github.com/DemisEom/SpecAugment

    Args:
        x: image
        y: label
        p: probability of performing operation
        frequency_masking_para: size of masked frequency region (number of sequential bins to mask)
        time_masking_para: size of masked time region (number of sequential windows to mask)
        frequency_mask_num: num of frequency regions to mask
        time_mask_num: number of time regions to mask

    returns:
        x: transformed features
        y: labels
    """

    aug = spec_augmentor(**kwargs)

    for i in range(len(x)):
        x[i] = aug(x[i])

    return x, y


def random_brightness(x, y, **kwargs):
    """
    Applied a random brightness augmentation to each feature.

    Args:
        x (numpy array): features
        y (numpy array): labels

    Returns:
        x: transformed features
        y: labels
    """
    aug = random_brighten(**kwargs)

    for i in range(len(x)):
        x[i] = aug(x[i])

    return x, y


### Albumentations augmentations ###
### https://albumentations.readthedocs.io/en/latest/index.html ###
def random_rotate(x, y, **kwargs):
    """
    Perfroms a random rotation using the albumentation rotate implementation.

    Args:
        x (numpy array): features
        y (numpy array): labels
        kwargs (keyword dict): See albumentations docs for kwargs:
                               https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Rotate

    Returns:
        x: transformed features
        y: labels
    """

    aug = Rotate(**kwargs)

    for i in range(len(x)):
        x[i] = aug(image=x[i])['image']

    return x, y


def elastic_transform(x, y, **kwargs):
    """
    Perfroms elastic transform using the albumentation implementation.

    Args:
        x (numpy array): features
        y (numpy array): labels
        kwargs (keyword dict): See albumentations docs for kwargs:
                               https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ElasticTransform

    Returns:
        x: transformed features
        y: labels
    """

    aug = ElasticTransform(**kwargs)

    for i in range(len(x)):
        x[i] = aug(image=x[i])['image']

    return x, y


def grid_distortion(x, y, **kwargs):
    """
    Perfroms grid distortion using the albumentation implementation.

    Args:
        x (numpy array): features
        y (numpy array): labels
        kwargs (keyword dict): See albumentations docs for kwargs:
                               https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDistortion

    Returns:
        x: transformed features
        y: labels
    """
    aug = GridDistortion(**kwargs)

    for i in range(len(x)):
        x[i] = aug(image=x[i])['image']

    return x, y


### Keras Augmentations ###
def image_augmentation(x, y, **kwargs):
    """
    Applies a random transformation to each image or spectrogram in X.
    See https://keras.io/preprocessing/image/ for a list of arguments (only transformation
    arguments apply)
    NOTE: modifies x in place (online) to save memory. If you need a copy call np.copy() before this function

    NOTE: Brightness range does not work because Keras scales spectrogram to an image by default

    Args:
        x: features
        y: labels
        **kwargs: parameter dict for Keras Image Processing constructor arguments

    Returns:
        x: transformed features
        y: optionally transformed labels
    """
    if "brightness_range" in kwargs:
        raise ValueError("Brightness Range not supported due to Keras scaling the spectrogram to image values")

    datagen = ImageDataGenerator(**kwargs)

    for i in range(len(x)):
        x[i] = datagen.random_transform(x[i])

    return x, y


def multi_augmentation(x, y, **kwargs):
    """
    Applies multiple augmentations (defined in the configuration file) to a single image.

    Args:
        x: features
        y: labels
        **kwargs (dict): Dict should contain a key "aug_funcs" which contains a list of 2 element lists [function location string, datamodel dict],
                         also includes optional parameter "shuffle_funcs" to randomize function order on each input image (defaults to true).

    Returns:
        x: transformed features
        y: optionally transformed labels
    """

    aug_funcs = kwargs['aug_funcs']     # list of 2 element lists [function location string, datamodel dict]
    shuffle_funcs = kwargs.get('shuffle_funcs', True)
    assert isinstance(shuffle_funcs, bool), f'shuffle_funcs parameter should be a boolean, not: {type(shuffle_funcs)}'

    indexes = np.arange(len(aug_funcs))
    if shuffle_funcs:
        np.random.shuffle(indexes)

    for i in indexes:
        func_str, params = aug_funcs[i]
        func = helper.get_method_from_string(func_str)

        x, y = func(x, y, **params)

    return x, y


# only for testing
def broken_augmentation(x, y, **kwargs):
    """
    Only for testing augmentation code. Randomizes class labels

    Args:
        x: features
        y: labels
        **kwargs (dict): Dict should contain a key "aug_funcs" which contains a list of 2 element lists [function location string, datamodel dict]

    Returns:
        x: transformed features
        y: optionally transformed labels
    """

    x.fill(1.0)
    return x, y
