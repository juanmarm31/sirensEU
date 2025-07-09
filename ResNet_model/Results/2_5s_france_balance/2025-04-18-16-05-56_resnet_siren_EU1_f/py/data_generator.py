import os
import warnings
import abc

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from tensorflow.keras.utils import Sequence

import numpy as np
try:
    from ..core import extract_features
    from ..logging_alf import getLogger, LogfileWriter, Level
except:
    from ALFpaka.core import extract_features
    from ALFpaka.logging_alf import getLogger, LogfileWriter, Level

__author__ = 'David Johnson'
__copyright__ = "Fraunhofer IDMT"


class AbstractDataGen(Sequence):

    def __init__(self, x, y, input_shape, input_dtype, augmentation_function=None, augmentation_parameters=None,
                 aug_factor=0, include_orig_data=True, batch_size=32, shuffle=True, eval=False):
        """
            Attributes:
                filelist (numpy array)  : array with a list of files for loading during each batch
                y (numpy array)         : classification labels
                batch_size (int)        : the size of each batch
                N (int)                 : total number of samples
                shape (int, int, int)   :  shape of the data

            Methods:
                __len__()            : returns the length / number of batches of the sequence
                __get_item__(index)  : returns the batch at index
        """

        if not isinstance(aug_factor, int):
            LOGGER.warning(
                'Aug factor is not an integer. This generator only supports integer factors, factor will be rounded down.')

        self._x = x
        self._y = y

        if eval is True:
            self._augmentation = None
            self._aug_factor = 1
        else:
            self._aug_factor = int(aug_factor)
            self._augmentation = augmentation_function

        self._augmentation_params = augmentation_parameters if augmentation_parameters is not None else {}

        if include_orig_data is False and self._aug_factor == 0:
            LOGGER.error('Aug Factor cannot be zero if original data is not included')
            raise ValueError('Aug Factor cannot be zero if original data is not included')

        self._include_orig_data = include_orig_data

        self.batch_size = batch_size

        # round number of iterations
        self._iterations = int(np.ceil(len(self._x) / self.batch_size))
        # get size of last batch (rest)
        self._last_batch_size = len(self._x) % self.batch_size

        # or full batch if nothing is left
        if self._last_batch_size == 0:
            self._last_batch_size = self.batch_size

        if self._augmentation is not None:
            if include_orig_data:
                self._real_batch_size = self.batch_size * (1 + self._aug_factor)
                self._last_real_batch_size = self._last_batch_size * (1 + self._aug_factor)
            else:
                self._real_batch_size = self.batch_size * self._aug_factor
                self._last_real_batch_size = self._last_batch_size * self._aug_factor
        else:
            self._real_batch_size = self.batch_size
            self._last_real_batch_size = self._last_batch_size

        self.N = len(self._x)
        self.shape = (self.N, *input_shape)

        self._shuffle = shuffle
        self._indexes = None

        # allocate final memory for each batch, take initial data type
        self._batch_x = np.zeros((self._real_batch_size, *input_shape), dtype=input_dtype)
        self._last_batch_x = np.zeros((self._last_real_batch_size, *input_shape), dtype=input_dtype)
        # for y take float32 since some augmentation interpolate labels and 32bit should be enough
        if self._y is not None:
            self._batch_y = np.zeros((self._real_batch_size, *y.shape[1:]), dtype=np.float32)
            self._last_batch_y = np.zeros((self._last_real_batch_size, *y.shape[1:]), dtype=np.float32)

        # init indexes
        self.on_epoch_end()

    @abc.abstractmethod
    def _get_batch(self, indexes, target_batch_x, target_batch_y):
        """
            Generates data containing batch_size samples
        """
        pass

    def __len__(self):
        """The number of batches per epoch"""
        return self._iterations

    def __getitem__(self, index):
        """Generate a batch of data at given index"""
        if index >= self._iterations:
            raise ValueError(f'Asked for Index {index} but Sequence has length {self._iterations}')

        # full batch
        if index < self._iterations - 1:
            index_array = self._indexes[index * self.batch_size:(index + 1) * self.batch_size]
            return self._get_batch(index_array, self._batch_x, self._batch_y)
        else:  # last batch only data left
            index_array = self._indexes[index * self.batch_size:]
            return self._get_batch(index_array, self._last_batch_x, self._last_batch_y)

    def on_epoch_end(self):
        """Runs after each epoch"""
        # updates indexes after each epoch
        self._indexes = np.arange(len(self._x))
        if self._shuffle:
            np.random.shuffle(self._indexes)

    def __iter__(self):
        """Create a python iterator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


class NumpyDataGen(AbstractDataGen):
    """
        A Keras Sequence/Iterator for generating batches of data from a Numpy Array
        in which features have already been loaded.
    """

    def __init__(self, x, y, augmentation_function=None, augmentation_parameters=None,
                 aug_factor=0, include_orig_data=True, batch_size=32, shuffle=True, eval=False):
        """
            Attributes:
                x (numpy array)         : array with loaded features
                y (numpy array)         : classification labels
                batch_size (int)        : the size of each batch
                N (int)                 : total number of samples
                shape (int, int, int)   :  shape of the data

            Methods:
                __len__()            : returns the length / number of batches of the sequence
                __get_item__(index)  : returns the batch at index
        """
        input_shape = x.shape[1:]
        input_dtype = x.dtype
        super(NumpyDataGen, self).__init__(x, y, input_shape, input_dtype, augmentation_function, augmentation_parameters, aug_factor,
                         include_orig_data, batch_size, shuffle, eval)


    def _get_batch(self, indexes, target_batch_x, target_batch_y):
        """
            Generates data containing batch_size samples
        """
        batch_size = len(indexes)

        batch_x = self._x[indexes]
        batch_y = self._y[indexes] if self._y is not None else None

        if self._augmentation is not None:
            if self._include_orig_data:  # copy original data to output
                end_idx = batch_size
                target_batch_x[0:end_idx] = np.copy(batch_x)
                if batch_y is not None:
                    target_batch_y[0:end_idx] = np.copy(batch_y)

            # augment
            for i in range(self._aug_factor):
                if self._include_orig_data:
                    start_idx = (i + 1) * batch_size
                else:
                    start_idx = i * batch_size

                end_idx = start_idx + batch_size

                # all augmentation is done inplace so we just copy batch to correct place and augment inplace
                target_batch_x[start_idx:end_idx] = np.copy(batch_x)
                if batch_y is not None:
                    target_batch_y[start_idx:end_idx] = np.copy(batch_y)
                    self._augmentation(target_batch_x[start_idx:end_idx], target_batch_y[start_idx:end_idx],
                                       **self._augmentation_params)
                else:
                    self._augmentation(target_batch_x[start_idx:end_idx], None, **self._augmentation_params)
        else:  # only original data (other else for no data is catched above)
            target_batch_x = np.copy(batch_x)
            if batch_y is not None:
                target_batch_y = np.copy(batch_y)

        # todo check for correct type conversion
        return target_batch_x, target_batch_y


class FileDataGen(AbstractDataGen):
    """
        A Keras Sequence/Iterator for generating batches of data from a filelist of numpy files
        with extract features
    """

    def __init__(self, filelist, y, augmentation_function=None,
                 augmentation_parameters=None, aug_factor=0, include_orig_data=True, batch_size=32, shuffle=True,
                 eval=False):
        """
            Attributes:
                filelist (numpy array)  : array with a list of files for loading during each batch
                y (numpy array)         : classification labels
                batch_size (int)        : the size of each batch
                N (int)                 : total number of samples
                shape (int, int, int)   :  shape of the data

            Methods:
                __len__()            : returns the length / number of batches of the sequence
                __get_item__(index)  : returns the batch at index
        """

        path = filelist[0][0]
        root, ext = os.path.splitext(path)
        self._filetype = ext
        if ext == '.npy':
            sample = np.load(path)
        elif ext == '.npz':
            self._load_y = True
            with np.load(path) as data:
                sample = data['x']
        else:
            raise ValueError(f'Invalid File Type - {ext}. Should be one of .npy or .npz')

        input_shape = sample.shape
        input_dtype = sample.dtype
        super(FileDataGen, self).__init__(filelist, y, input_shape, input_dtype, augmentation_function, augmentation_parameters, aug_factor,
                         include_orig_data, batch_size, shuffle, eval)


    def _get_batch(self, indexes, target_batch_x, target_batch_y):
        """Generates data containing batch_size samples
        """
        batch_size = len(indexes)

        # load files and sve to a batch
        batch_x = []
        for f in self._x[indexes]:
            if self._filetype == '.npy':
                tmp_x = np.load(f[0])
            else:
                with np.load(f[0]) as data:
                    with np.load(f[0]) as data:
                        tmp_x = data['x']
            batch_x.append(tmp_x)

        batch_x = np.array(batch_x)
        batch_y = self._y[indexes] if self._y is not None else None     # y is always loaded from ALF input

        # augment batch
        if self._augmentation is not None:
            if self._include_orig_data:  # copy original data to output
                end_idx = batch_size
                target_batch_x[0:end_idx] = np.copy(batch_x)
                if batch_y is not None:
                    target_batch_y[0:end_idx] = np.copy(batch_y)

            # augment
            for i in range(self._aug_factor):
                if self._include_orig_data:
                    start_idx = (i + 1) * batch_size
                else:
                    start_idx = i * batch_size

                end_idx = start_idx + batch_size

                # all augmentation is done inplace so we just copy batch to correct place and augment inplace
                target_batch_x[start_idx:end_idx] = np.copy(batch_x)
                if batch_y is not None:
                    target_batch_y[start_idx:end_idx] = np.copy(batch_y)
                    self._augmentation(target_batch_x[start_idx:end_idx], target_batch_y[start_idx:end_idx],
                                       **self._augmentation_params)
                else:
                    self._augmentation(target_batch_x[start_idx:end_idx], None, **self._augmentation_params)
        else:  # only original data (other else for no data is caught above)
            target_batch_x = np.copy(batch_x)
            if batch_y is not None:
                target_batch_y = np.copy(batch_y)

        # todo check for correct type conversion
        return target_batch_x, target_batch_y


# LOCAL GENERATOR TESTS
def loadconfig_test(config, expected_gen_params):
    mainParams, trainParams, evalParams = read_training_config.readTrainingConfig(config, "TESTTEMP", "TESTOUTPUT")

    assert mainParams.generator_train_params == expected_gen_params, 'Generator parameters did not load correctly'
    dg = mainParams.generator
    assert dg == generators.data_generator.NumpyDataGen, 'Data Generator not loaded properly'
    LOGGER.debug('TEST: LOAD CONFIG PASSES')


def loaddata_test(config):
    mainParams, trainParams, evalParams = read_training_config.readTrainingConfig(config, "TESTTEMP", "TESTOUTPUT")
    mainParams.class_id_dict = {"class1": 0, "class2": 2, "class3": 3} # hack for now
    allFeatures = extract_features.extractAllFeatures(main_parameters=mainParams, train_parameters=trainParams, mt=False)
    featStackedTrain, class_idStackedTrain, framesPerFileTrain, dataFilesNamesTrain, augmentedFeatTrain, augmentedClassIdsTrain = get_features_for_dataset.getFeaturesAndClassIDsFromDataset(
        allFeatures['train'], addAugmentedData=True)

    dg = mainParams.generator
    dg = dg(featStackedTrain, class_idStackedTrain, **mainParams.generator_train_params)

    expected_size = featStackedTrain.shape[0]
    assert dg.N == expected_size, f'Generator not loading correct amount of data. Expected {expected_size} : Actual {dg.N}'
    assert len(dg) == np.ceil(featStackedTrain.shape[0] / mainParams.generator_train_params['batch_size']), 'Generator not loading all the data'

    aug_factor = mainParams.generator_train_params['aug_factor'] if 'aug_factor' in mainParams.generator_train_params else 0
    include_orig = mainParams.generator_train_params['include_orig_data'] if 'include_orig_data' in mainParams.generator_train_params else True

    default_batch_size = mainParams.generator_train_params['batch_size']
    last_batch_size = featStackedTrain.shape[0] % mainParams.generator_train_params['batch_size']

    if aug_factor > 0:
        if include_orig:
            default_batch_size *= aug_factor
            last_batch_size *= aug_factor
        else:
            default_batch_size *= aug_factor
            last_batch_size *= aug_factor

    for i, (x, y) in enumerate(dg):
        if i < len(dg) - 1:
            expsize = default_batch_size
        else:
            expsize = last_batch_size

        assert x.shape[
                   0] == expsize, f'Incorrect number of features in batch {i}. Expected: {expsize} | Actual: {x.shape[0]}'
        assert y.shape[
                   0] == expsize, f'Incorrect number of labels in batch {i}. Expected: {expsize} | Actual: {y.shape[0]}'
        assert x.shape[1:] == (
            15, 3, 1), f'Incorrect feature size in batch {i}. Expected: (15, 3, 1) | Actual: {x.shape[1:-1]}'

    LOGGER.debug('TEST: LOADING DATA PASSES')


def augmentation_test(config):
    from ALFpaka.generators import general_augmentations

    mainParams, trainParams, evalParams = read_training_config.readTrainingConfig(config, "TESTTEMP", "TESTOUTPUT")

    assert mainParams.generator_train_params['augmentation_function'] == general_augmentations.mixup, 'Augmentation Function did not load correctly'
    assert mainParams.generator_train_params['augmentation_parameters'] == {'alpha': 0.4}, 'Augmentation parameters did not load correctly'

    mainParams.class_id_dict = {"class1": 0, "class2": 2, "class3": 3} # hack for now
    allFeatures = extract_features.extractAllFeatures(main_parameters=mainParams, train_parameters=trainParams, mt=False)
    featStackedTrain, class_idStackedTrain, framesPerFileTrain, dataFilesNamesTrain = get_features_for_dataset.getFeaturesAndClassIDsFromDataset(
        allFeatures['train'], addAugmentedData=False)
    idMatTrain, num_classes = helper_keras.changeLabelVectorToMatrix(class_idStackedTrain, get_datasets.getNumberOfClassesFromClassDict(mainParams.class_id_dict))

    dg = mainParams.generator

    mainParams.generator_train_params['shuffle'] = False  # set to false for easier testing
    dg_iter_mixup = dg(featStackedTrain, idMatTrain, **mainParams.generator_train_params)
    mainParams.generator_train_params['include_orig_data'] = True
    dg_iter_noaug = dg(featStackedTrain, idMatTrain, **mainParams.generator_eval_params)

    assert featStackedTrain.shape[1:] == dg_iter_mixup.shape[ 1:], f'Augmented Data is not the same shape as the original data, but should be since original data is not included and aug factor is 1'
    assert len(dg_iter_mixup) == len(dg_iter_noaug), f'Aug and NoAug gens should be same length since aug_factor=1 and original data is not included. Aug Len={len(dg_iter_mixup)} : NoAug Len={len(dg_iter_noaug)}'

    LOGGER.debug('TEST: AUGMENTATION CONFIG LOADING PASSES')


def augmentation_with_increase_test(config):
    from ALFpaka.generators import general_augmentations

    mainParams, trainParams, evalParams = read_training_config.readTrainingConfig(config, "TESTTEMP", "TESTOUTPUT")

    assert mainParams.generator_train_params['augmentation_function'] == general_augmentations.mixup, 'Augmentation Function did not load correctly'
    assert mainParams.generator_train_params['augmentation_parameters'] == {'alpha': 0.4}, 'Augmentation parameters did not load correctly'

    dg = mainParams.generator
    assert dg == generators.data_generator.NumpyDataGen, 'Data Generator not loaded properly'

    mainParams.class_id_dict = {"class1": 0, "class2": 2, "class3": 3} # hack for now
    allFeatures = extract_features.extractAllFeatures(main_parameters=mainParams, train_parameters=trainParams, mt=False)
    featStackedTrain, class_idStackedTrain, framesPerFileTrain, dataFilesNamesTrain = get_features_for_dataset.getFeaturesAndClassIDsFromDataset(
        allFeatures['train'], addAugmentedData=False)
    idMatTrain, num_classes = helper_keras.changeLabelVectorToMatrix(class_idStackedTrain, get_datasets.getNumberOfClassesFromClassDict(mainParams.class_id_dict))

    dg_iter_mixup = dg(featStackedTrain, idMatTrain, **mainParams.generator_train_params)

    assert featStackedTrain.shape[1:] == dg_iter_mixup.shape[1:], f'Augmented features is not the same shape as the original data'

    aug_factor = mainParams.generator_train_params['aug_factor'] if 'aug_factor' in mainParams.generator_train_params else 0
    include_orig = mainParams.generator_train_params['include_orig_data'] if 'include_orig_data' in mainParams.generator_train_params else True

    default_batch_size = mainParams.generator_train_params['batch_size']
    last_batch_size = featStackedTrain.shape[0] % mainParams.generator_train_params['batch_size']

    if aug_factor > 0:
        if include_orig:
            default_batch_size *= (aug_factor + 1)
            last_batch_size *= (aug_factor + 1)
        else:
            default_batch_size *= aug_factor
            last_batch_size *= aug_factor

    for i, (mixup, mx_y) in enumerate(dg_iter_mixup):
        if i < len(dg_iter_mixup) - 1:
            expsize = default_batch_size
        else:
            expsize = last_batch_size
        assert mixup.shape[
                   0] == expsize, f'Augmented Batch Size not correct on iteration {i}. Expected: {expsize} : Actual {len(mixup)}'

    LOGGER.debug('TEST: AUGMENTATION WITH INCREASED DATA CONFIG LOADING PASSES')


def main():
    repo_path = Path(__file__).parent.parent
    CONFIG = repo_path / 'core/test/configs/test_train_config_cnn_generator.json'
    AUG_CONFIG = repo_path / 'core/test/configs/test_train_config_cnn_generatorwithaug.json'
    AUGINC_CONFIG = repo_path / 'core/test/configs/test_train_config_cnn_generatorwithaugincrease.json'

    try:
        loadconfig_test(CONFIG, {'batch_size': 64, 'shuffle': True})
        loaddata_test(CONFIG)
        loadconfig_test(AUG_CONFIG, {'aug_factor': 1, 'include_orig_data': False, 'batch_size': 64, 'shuffle': True,
                                     'augmentation_function' : ALFpaka.generators.general_augmentations.mixup, 'augmentation_parameters' : {'alpha': 0.4}})
        augmentation_test(AUG_CONFIG)
        augmentation_with_increase_test(AUGINC_CONFIG)
    finally:
        if os.path.exists('TESTTEMP'):
            shutil.rmtree('TESTTEMP')

if __name__ == '__main__':
    import shutil
    from pathlib import Path
    import ALFpaka.generators.general_augmentations
    from ALFpaka.core.config import read_training_config
    from ALFpaka.core import get_datasets
    from ALFpaka.core import get_features_for_dataset
    from ALFpaka.core.helper import helper_keras
    import ALFpaka.generators as generators
    import ALFpaka.logging_alf as logging

    LOGGER = logging.logfile_writer.LogfileWriter(name=__file__, console_log_level=logging.Level.DEBUG,
                           log_file_level=None, explicit_error_log_file=False).get_logger()
    main()
else:
    LOGGER = getLogger()
