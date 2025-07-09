"""
Created on 05.02.2021

@author: goh
"""

import librosa
import numpy as np
import os
from ALFpaka.core.helper import helper, extraction_tools
from ALFpaka.logging_alf import getLogger

LOGGER = getLogger()

FFT_SIZE_DEFAULT = 2048


def extractSTFT(file, class_id, **extract_params):
    """Extract magnitude or mel spectrogram.

        Valid parameters are:
            fftsize = fftsize for STFT transformation (default: 2048)
            hopsize = hopsize for STFT transformation (default: winsize/2)
            log_spec = logarithmic scaling of spectrogram (default: True)
            mel_bands = number of mel bands, normal magn spec if 0 (default: 0)
            power = power of value for spectrogramm, 1 = normal spectrogram, 2 = power spectrogram (default: 2)
            mono = if true channels will be converted to mono (default: True)
            sample_rate (deprecated desired_sample_rate) = if set change sample rate of audio (default: None)
            pcen = apply pcen with default parameters (default: False) - will be done instead of log_spec.
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
            pitch_shift = list with pitch shifts to augment training data in semi tones [-2, 1, 1, 2]
            time_stretch = list with time stretch factors to augment test data in % [0.9, 0.95, 1.05, 1.1]
            num_augment = number of augmentations, if set num will be randomly picked from pitch/time
                            otherwise do all pitch and time augmentations (default: None)
            minimum_duration = files will be zero padded if shorter than minimum duration in seconds
            average = return mean and std of features
            dtype = string with dtype of features (default: "float32")
        Args:
            file (string or numpy array): file path to audio or numpy file or numpy arrays with shape [samples, channel]
            class_id (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    fftsize = extract_params.get("fftsize", FFT_SIZE_DEFAULT)
    hopsize = extract_params.get("hopsize", int(fftsize / 2))
    mel_bands = extract_params.get("mel_bands", 0)
    power = extract_params.get("power", 2)
    log_spec = extract_params.get("log_spec", True)
    average = extract_params.get("average", False)
    # legacy log spec (used before 01.09.2022) with librosa db scale where each spectrogram was normalized by itself - this is removed as well as limit of -80db
    # use only to replicate previous results
    normalized_log_spec = extract_params.get("normalized_log_spec", False)
    dtype = extract_params.get("dtype", "float32")
    if dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    elif dtype == "float16":
        dtype = np.float16
    else:
        raise RuntimeError("Unknown dtype")

    # deprecated but for compatibility
    if "logSpec" in extract_params:
        log_spec = extract_params["logSpec"]
    mono = extract_params.get("mono", True)

    # for numpy we need initial sample rate
    initial_sample_rate = extract_params.get("initial_sample_rate", None)

    desired_sample_rate = extract_params.get("desired_sample_rate", None)
    # updated for better parameter name
    desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)
    normalize_audio_param = extract_params.get("normalize_audio", False)
    pcen = extract_params.get("pcen", False)

    pitch_shift = extract_params.get("pitch_shift", None)
    time_stretch = extract_params.get("time_stretch", None)
    num_augment = extract_params.get("num_augment", None)

    min_duration = extract_params.get("minimum_duration", 0)
    if pcen is True and log_spec is True:
        LOGGER.warning("Log_spec ignored when pcen is set")

    # support for numpy arrays with shape [samples, channel]
    if not isinstance(file, str):
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy arrays")
        y = file
        if desired_sample_rate is None:
            sr = initial_sample_rate
        else:
            y = librosa.resample(y=y, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
    elif file.endswith(".npy"):  # support for numpy files
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy files")
        y = np.load(file)
        if desired_sample_rate is not None and initial_sample_rate != desired_sample_rate:
            y = librosa.resample(y=y, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
        else:
            sr = initial_sample_rate
    else:  # audio
        y, sr = librosa.load(file, sr=desired_sample_rate, mono=mono, res_type='scipy')

    if pcen is True:
        power = 1

    if normalize_audio_param is True:
        helper.normalize_audio(samples=y)

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    # pad files with zeros if they should have minimum duration
    if min_duration > 0:
        min_duration = int(min_duration * sr + 0.5)

        if y.shape[1] < min_duration:
            y_new = np.zeros((y.shape[0], min_duration), dtype=y.dtype)
            y_new[:, :y.shape[1]] = y
            y = y_new

    y_all = [y]

    # pitch shift & time stretch
    if pitch_shift is not None or time_stretch is not None:
        # do all
        if num_augment is None:
            if pitch_shift is not None:
                for p in pitch_shift:
                    y_aug = []
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.pitch_shift(y[channel], sr=sr, n_steps=p))
                    y_all.append(np.asarray(y_aug))
            if time_stretch is not None:
                for t in time_stretch:
                    y_aug = []
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.time_stretch(y[channel], rate=t))
                    y_all.append(np.asarray(y_aug))
        else:  # pick only few
            # reseeding for randomization in multiprocessing
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            for _ in range(num_augment):
                # pick pitch shift or time stretch
                if time_stretch is not None and pitch_shift is not None:
                    p = (np.random.uniform() > 0.5)
                elif time_stretch is None:
                    p = True
                else:
                    p = False

                # and select random value
                y_aug = []
                if p:
                    pitch = np.random.choice(pitch_shift, size=1)[0]
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.pitch_shift(y[channel], sr=sr, n_steps=pitch))
                else:
                    stretch = np.random.choice(time_stretch, size=1)[0]
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.time_stretch(y[channel], rate=stretch))
                y_all.append(np.asarray(y_aug))

    all_x = []
    for y in y_all:
        # extract stft for all channels and add to "color" channel (last index)
        all_channels_x = []
        for l in range(y.shape[0]):
            if mel_bands > 0:
                x = librosa.feature.melspectrogram(y=y[l], sr=sr, n_fft=fftsize, hop_length=hopsize, n_mels=mel_bands,
                                                   power=power)
            else:
                x, _ = librosa.magphase(librosa.stft(y=y[l], n_fft=fftsize, hop_length=hopsize), power=power)

            if pcen is True:
                x = librosa.pcen(x * (2 ** 31), sr=sr, hop_length=hopsize)
                # first frame due to initialisation is off so we replicate the second
                x[:, 0] = x[:, 1]
            elif log_spec is True:
                x = 10.0 * (np.log10(x + 1e-7))  # add epsilon to avoid log(0)
            elif normalized_log_spec is True:  # legacy version - use only to replicate results
                x = librosa.power_to_db(x, ref=np.max)

            x = x.T
            all_channels_x.append(x)
        x = np.asarray(all_channels_x)
        # move channels to last dimension (color channel)
        x = np.moveaxis(x, 0, -1)

        if x.shape[-1] == 1:
            x = np.squeeze(x)

        all_x.append(x)

    # get non augmented one
    x_orig = all_x.pop(0)
    if average:
        feat_x_mean = np.mean(x_orig, axis=0, keepdims=True)
        feat_x_std = np.std(x_orig, axis=0, keepdims=True)
        x_orig = np.hstack((feat_x_mean, feat_x_std))
    classes_orig = np.ones(x_orig.shape[0]) * class_id

    # nothing augmented
    if len(all_x) == 0:
        return [x_orig.astype(np.float32), classes_orig.astype(np.uint16), file]

    x_aug = np.vstack(all_x)
    if average:
        feat_x_mean = np.mean(x_aug, axis=0, keepdims=True)
        feat_x_std = np.std(x_aug, axis=0, keepdims=True)
        x_aug = np.hstack((feat_x_mean, feat_x_std))
    classes_aug = np.ones(x_aug.shape[0]) * class_id

    return [x_orig.astype(dtype), classes_orig.astype(np.uint16), file, x_aug.astype(dtype),
            classes_aug.astype(np.uint16)]


def extractSTFT2D(file, class_id, **extract_params):
    """Extract magnitude or mel spectrogram for CNN.

        Valid parameters are:
            fftsize = fftsize for STFT transformation (default: 2048)
            hopsize = hopsize for STFT transformation (default: winsize/2)
            log_spec = logarithmic scaling of spectrogram (default: True)
            mel_bands = number of mel bands, normal magn spec if 0 (default: 0)
            mono: if true channels will be converted to mono (default: True)
            desired_sample_rate = if set change sample rate of audio (default: None)
            pcen = apply pcen with default parameters (default: False) - will be done instead of log_spec.
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
            patchsize = number of time frames per CNN patch
            patchsize_in_seconds = set CNN patch size in seconds. Mutually exclusive use with 'patchsize' (default: None - patchsize in time frames used)
            patchhop = hppsize for creating patches (default: patchsize - no overlap)
            patchhop_in_seconds = set overlaping of patches in seconds. Mutually exclusive use with 'patchhop' (default: patchsize_in_seconds)
            pitch_shift = list with pitch shifts to augment training data in semi tones [-2, 1, 1, 2]
            time_stretch = list with time stretch factors to augment test data in % [0.9, 0.95, 1.05, 1.1]
            num_augment = number of augmentations, if set num will be randomly picked from pitch/time
                            otherwise do all pitch and time augmentations (default: None)
            minimum_duration = files will be zero padded if shorter than minimum duration in seconds
            dtype = string with dtype of features (default: "float32")
        Args:
            file (string): string of file path
            class_id (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    dtype = extract_params.get("dtype", "float32")
    if dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    elif dtype == "float16":
        dtype = np.float16
    else:
        raise RuntimeError("Unknown dtype")

    patchsize = extract_params.get("patchsize", None)
    patchsize_in_seconds = extract_params.get("patchsize_in_seconds", None)

    if patchsize_in_seconds is not None and patchsize is not None:
        LOGGER.error(
            "Use either 'patchsize' or 'patchsize_in_seconds' when defining the feature extraction function in config file")
        raise RuntimeError

    res = extractSTFT(file, class_id, **extract_params)

    if patchsize_in_seconds is not None:
        # for conversion (bit complicated since this information is normally extracted with STFT
        fftsize = extract_params.get("fftsize", FFT_SIZE_DEFAULT)
        hopsize = extract_params.get("hopsize", int(fftsize / 2))
        desired_sample_rate = extract_params.get("desired_sample_rate", None)

        # updated for better parameter name
        desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)

        if desired_sample_rate is None:
            LOGGER.error("desired_sample_rate must be set when patch size is set in seconds")
            raise ValueError("desired_sample_rate must be set when patch size is set in seconds")

        patchhop_in_seconds = extract_params.get("patchhop_in_seconds", patchsize_in_seconds)

        # librosa stft is centered (and padded) so full file can be used
        patchsize = int((patchsize_in_seconds * desired_sample_rate / hopsize) + 1)
        patchhop = int((patchhop_in_seconds * desired_sample_rate / hopsize) + 1)
    else:
        # take full patch
        if patchsize is None:
            patchsize = res[0].shape[0]
        patchhop = extract_params.get("patchhop", patchsize)

    patches = extraction_tools.extract_2D_patches_from_features(res[0], patchsize, patchhop)
    classes = np.ones(patches.shape[0]) * class_id

    # not augmented
    if len(res) < 5:
        return [patches.astype(res[0].dtype), classes.astype(np.uint16), file]

    # take patches from full augmented (overlapt between files shoulnd't hurt since it's the same class)
    patches_aug = extraction_tools.extract_2D_patches_from_features(res[3], patchsize, patchhop)
    classes_aug = np.ones(patches_aug.shape[0]) * class_id
    return [patches.astype(res[0].dtype), classes.astype(np.uint16), file, patches_aug.astype(dtype),
            classes_aug.astype(np.uint16)]


def extractCQT(file, class_id, **extract_params):
    """Extract cqt.

        Valid parameters are:
            hopsize = hopsize for STFT transformation (default: winsize/2)
            bins_per_octave: number of bins per octave (default: 12)
            n_octaves: number of octaves to cover (default: 7)
            fmin: start cqt at this frequency in Hz (default: 50)
            mono: if true channels will be converted to mono (default: True)
            desired_sample_rate = if set change sample rate of audio (default: None)
            logSpec = logarithmic scaling of spectrogram (default: True)
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
        Args:
            file (string): string of file path
            classID (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    hopsize = extract_params.get("hopsize", 1024)
    bins_per_octave = extract_params.get("bins_per_octave", 12)
    n_octaves = extract_params.get("n_octaves", 7)
    mono = extract_params.get("mono", True)
    desired_sample_rate = extract_params.get("desired_sample_rate", None)
    desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)
    normalize_audio_param = extract_params.get("normalize_audio", False)
    log_spec = extract_params.get("log_spec", True)
    # deprecated but for compatibility
    if "logSpec" in extract_params:
        log_spec = extract_params["logSpec"]
    fmin = extract_params.get("fmin", 50)

    y, sr = librosa.load(file, sr=desired_sample_rate, mono=mono, res_type='scipy')

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    if normalize_audio_param is True:
        helper.normalize_audio(samples=y)

    all_channels_x = []
    for l in range(y.shape[0]):
        x = np.abs(librosa.cqt(y[l], sr=sr, hop_length=hopsize, bins_per_octave=bins_per_octave,
                               n_bins=n_octaves * bins_per_octave, fmin=fmin))

        if log_spec:
            x = librosa.amplitude_to_db(x, ref=np.max)

        x = x.T
        all_channels_x.append(x)
    x = np.asarray(all_channels_x)
    # move channels to last dimension (color channel)
    x = np.moveaxis(x, 0, -1)

    if x.shape[-1] == 1:
        x = np.squeeze(x)

    classes = np.ones(x.shape[0]) * class_id
    return [x.astype(np.float32), classes.astype(np.uint16), file]


def extractCQT2D(file, class_id, **extract_params):
    """Extract cqt for CNN.

        Valid parameters are:
            hopsize = hopsize for STFT transformation (default: winsize/2)
            bins_per_octave: number of bins per octave (default: 12)
            n_octaves: number of octaves to cover (default: 7)
            fmin: start cqt at this frequency in Hz (default: 50)
            mono: if true channels will be converted to mono (default: True)
            desired_sample_rate = if set change sample rate of audio (default: None)
            logSpec = logarithmic scaling of spectrogram (default: True)
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
            patchsize = number of time frames per CNN patch
            patchhop = hppsize for creating patches (default: patchsize - no overlap)
        Args:
            file (string): string of file path
            class_id (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    patchsize = extract_params.get("patchsize")
    patchhop = extract_params.get("patchhop", patchsize)

    x, _, _ = extractCQT(file, class_id, **extract_params)

    patches = extraction_tools.extract_2D_patches_from_features(x, patchsize, patchhop)
    classes = np.ones(patches.shape[0]) * class_id

    return [patches.astype(np.float32), classes.astype(np.uint16), file]


def extractRawAudio(file, class_id, **extract_params):
    """Extract audio data.

        Valid parameters are:
            mono: if true channels will be converted to mono (default: True)
            desired_sample_rate = if set change sample rate of audio (default: None)
            initial_sample_rate = needs to be set for numpy files
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
            snippet_length = in samples if audio should be cut to fixed length (default: None)
            snippet_length_in_seconds = in seconds if audio should be cut to fixed length (default: None)
            snippet_hop = hop length in samples for cut audio (default: None)
            snippet_hop_in_seconds = hop length in seconds for cut audio (default: None)
            cutoff_start = in samples length to cut audio at start, starting from sample 0 (default: 0)
            cutoff_start_in_seconds = in seconds length to cut audio at start, starting from 0 s (default: 0)
            cutoff_end = in samples length to cut audio at end, starting from sample 0 (default: 0)
            cutoff_end_in_seconds = in seconds length to cut audio at end, starting from 0 s (default: 0)
        Args:
            file (string): string of file path
            classID (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    mono = extract_params.get("mono", True)
    desired_sample_rate = extract_params.get("desired_sample_rate", None)
    desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)
    initial_sample_rate = extract_params.get("initial_sample_rate", None)
    normalize_audio = extract_params.get("normalize_audio", False)
    snippet_length = extract_params.get("snippet_length", None)
    snippet_length_in_seconds = extract_params.get("snippet_length_in_seconds", None)
    snippet_hop = extract_params.get("snippet_hop", None)
    snippet_hop_in_seconds = extract_params.get("snippet_hop_in_seconds", None)
    b_snippet_pad = extract_params.get("snippet_pad", False)
    cutoff_start = extract_params.get("cutoff_start", 0)
    cutoff_end = extract_params.get("cutoff_end", 0)
    cutoff_start_in_seconds = extract_params.get("cutoff_start_in_seconds", 0)
    cutoff_end_in_seconds = extract_params.get("cutoff_end_in_seconds", 0)

    if ("snippet_length" in extract_params) and ("snippet_length_in_seconds" in extract_params):
        LOGGER.error(
            "Error: use only one of the parameters 'snippet_length' or 'snippet_length_in_seconds' when defining the raw audio extraction function in config file")
        raise RuntimeError

    if ("snippet_hop" in extract_params) and ("snippet_hop_in_seconds" in extract_params):
        LOGGER.error(
            "Error: use only one of the parameters 'snippet_hop' or 'snippet_hop_in_seconds' when defining the raw audio function in config file")
        raise RuntimeError

    if ("cutoff_start" in extract_params) and ("cutoff_start_in_seconds" in extract_params):
        LOGGER.error(
            "Error: use only one of the parameters 'cutoff_start' or 'cutoff_start_in_seconds' when defining the raw audio extraction function in config file")
        raise RuntimeError

    if ("cutoff_end" in extract_params) and ("cutoff_end_in_seconds" in extract_params):
        LOGGER.error(
            "Error: use only one of the parameters 'cutoff_end' or 'cutoff_end_in_seconds' when defining the raw audio function in config file")
        raise RuntimeError

    # support for numpy files
    if file.endswith(".npy"):
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy files")
        x = np.load(file)
        if desired_sample_rate is not None and initial_sample_rate != desired_sample_rate:
            x = librosa.resample(y=y, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
        else:
            sr = initial_sample_rate
    else:
        x, sr = librosa.load(file, sr=desired_sample_rate, mono=mono, res_type='scipy')

    if not (cutoff_start_in_seconds == 0):
        cutoff_start = cutoff_start_in_seconds * sr
    if not (cutoff_end_in_seconds == 0):
        cutoff_end = cutoff_end_in_seconds * sr

    if cutoff_end == 0:
        cutoff_end = len(x)

    if (cutoff_start < 0) or (cutoff_end < 0):
        LOGGER.error("Error: Cutoff at start and cutoff at end should both be positive")
        raise RuntimeError

    if cutoff_start >= len(x):
        LOGGER.error("Error: Cutoff at start bigger or equal than file length. No signal will be taken into account")
        raise RuntimeError

    if cutoff_end > len(x):
        LOGGER.warning("Warning: Cutoff at end bigger or equal than file length. No signal will be cut at end")
        cutoff_end = len(x)

    if cutoff_end <= cutoff_start:
        LOGGER.error("Error: Cutoff at end smaller or equal to cutoff at start. No signal will be taken into account")

    x = x[cutoff_start:cutoff_end]

    if normalize_audio is True:
        helper.normalize_audio(samples=x)

    # bring mono to same shape as stereo
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)

    # put channels to end (similar to color)
    x = x.T

    if snippet_length is not None:
        snippet_length_in_seconds = snippet_length / sr
    if snippet_hop is not None:
        snippet_hop_in_seconds = snippet_hop / sr

    x = cut_audio(x, sr, snippet_length_in_seconds, snippet_hop_in_seconds, b_snippet_pad)

    classes = np.ones(x.shape[0]) * class_id

    return [x.astype(np.float32), classes.astype(np.uint16), file]


def extractTraditionalFeatures(file, class_id, **extract_params):
    """Extract audio dat.
         Valid parameters are:
             traditional_features = list of features to use - supported are "rmse", "centroid", "bandwidth", "flatness", "rolloff", "contrast", "mfcc"
             desired_sample_rate = if set change sample rate of audio (default: None)
             initial_sample_rate = needs to be set for numpy files
             normalize_audio = normalize each audio file to maximum of 1 (default: False)
             snippet_length = in samples if audio should be cut to fixed length (default: None)
             snippet_length_in_seconds = in seconds if audio should be cut to fixed length (default: None)
             snippet_hop = hop length in samples for cut audio (default: None)
             snippet_hop_in_seconds = hop length in seconds for cut audio (default: None)
             average = return mean and std of features
         Args:
             file (string): string of file path
             classID (dict): class id of file
             extract_params (**kwargs): dict with additional parameters (alf interface)
         Returns:
             [x, y, file name]
         """
    snippet_length = extract_params.get("snippet_length", None)
    snippet_length_in_seconds = extract_params.get("snippet_length_in_seconds", None)
    snippet_hop = extract_params.get("snippet_hop", None)
    snippet_hop_in_seconds = extract_params.get("snippet_hop_in_seconds", None)
    average = extract_params.get("average", True)
    normalize_audio = extract_params.get("normalize_audio", False)
    desired_sample_rate = extract_params.get("desired_sample_rate", None)
    desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)
    initial_sample_rate = extract_params.get("initial_sample_rate", None)
    traditional_features = extract_params.get("traditional_features",
                                              ["rmse", "centroid", "bandwidth", "flatness", "rolloff", "contrast",
                                               "mfcc"])

    # support for numpy files
    if file.endswith(".npy"):
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy files")
        x = np.load(file)
        if desired_sample_rate is not None and initial_sample_rate != desired_sample_rate:
            x = librosa.resample(y=x, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
        else:
            sr = initial_sample_rate
    else:
        x, sr = librosa.load(file, sr=desired_sample_rate, mono=True, res_type='scipy')

    if snippet_length is not None:
        snippet_length_in_seconds = snippet_length / sr
    if snippet_hop is not None:
        snippet_hop_in_seconds = snippet_hop / sr

    if normalize_audio is True:
        helper.normalize_audio(samples=x)

    x_cut = cut_audio(x, sr, snippet_length_in_seconds, snippet_hop_in_seconds, False)

    feat_all = []

    for l in range(x_cut.shape[0]):
        x = x_cut[l]
        feat_x = []
        if "rmse" in traditional_features:
            feat_x.append(librosa.feature.rms(y=x))
        if "centroid" in traditional_features:
            feat_x.append(librosa.feature.spectral_centroid(y=x, sr=sr))
        if "bandwidth" in traditional_features:
            feat_x.append(librosa.feature.spectral_bandwidth(y=x, sr=sr))
        if "flatness" in traditional_features:
            feat_x.append(librosa.feature.spectral_flatness(y=x))
        if "rolloff" in traditional_features:
            feat_x.append(librosa.feature.spectral_rolloff(y=x, sr=sr))
        if "contrast" in traditional_features:
            feat_x.append(librosa.feature.spectral_contrast(y=x, sr=sr))
        if "mfcc" in traditional_features:
            feat_x.append(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13))

        feat_x = np.concatenate(feat_x, axis=0).T

        # take mean and avg per snippet
        if average is True:
            feat_x_mean = np.mean(feat_x, axis=0, keepdims=True)
            feat_x_std = np.std(feat_x, axis=0, keepdims=True)
            feat_x = np.concatenate((feat_x_mean, feat_x_std), axis=1)

        feat_all.append(feat_x)

    feat_all = np.vstack(feat_all)
    classes = np.ones(feat_all.shape[0]) * class_id

    return [feat_all.astype(np.float32), classes.astype(np.uint16), file]


def cut_audio(x, sr, snippet_length_sec, snippet_hop_sec, b_snippet_pad=False):
    if snippet_length_sec is None:  # don't cut audio snippets
        x = np.expand_dims(x, axis=0)

    # cut audio to snippets, channels must be at end (samples, channel)
    else:
        if snippet_hop_sec is None:
            snippet_hop_sec = snippet_length_sec / 2

        snippet_length = int(sr * snippet_length_sec)  # change to samples
        snippet_hop = int(sr * snippet_hop_sec)  # change to samples

        if snippet_length > x.shape[0]:
            if b_snippet_pad is True:
                x_new = np.zeros((snippet_length, x.shape[1]))
                x_new[:x.shape[0], :] = x
                x = x_new
            else:
                raise RuntimeError("snippet_length must be set to minimum duration (padding was set to disabled)")

        num_snippets = int((x.shape[0] - (snippet_length - snippet_hop)) / snippet_hop)

        # only cut if it makes sense
        if num_snippets > 1:
            x_arr = []
            for l in range(num_snippets):
                idx = l * snippet_hop
                x_s = x[idx:idx + snippet_length]
                x_arr.append(x_s)
            x = np.asarray(x_arr)
        else:  # in case we did not cut it is one example per clip
            x = x[0:snippet_length]
            x = np.expand_dims(x, axis=0)

    return x


def extractMFCC(file, class_id, **extract_params):
    """Extract Mel Frequency Cepstral Coefficients (MFCC).
    How to compute the MFCCs: Waveform----> DFT ----> Log-Amplitude Spectrum ----> Mel-Scaling (mel filterbank) ----> Discrete Cosine Transform (DCT) ----> MFCCs
    Intution: MFCC derives the basic information about or construction of the spectrum by ignoring the details fo spectrum.In other word, MFCCs describe more the large structures of the Spectrum and ignor its fine structures.
              The DCT tries to fit cosine waves with different frequencies to the log-Spectrum and comes finally up with the values (coefficients) indicating how well they fit to the original log-Spectrum.
                The highest ranked coefficients keep most information about the spectrum, i.e rather representing the slower changing components of a spectrum than the fast changing ones.

        Valid parameters are:
            fftsize = fftsize for STFT transformation (default: 2048)
            hopsize = hopsize for STFT transformation (default: winsize/2)
            mono = if true channels will be converted to mono (default: True)
            sample_rate (deprecated desired_sample_rate) = if set change sample rate of audio (default: None)
            normalize_audio_param = normalize each audio file to maximum of 1 (default: False)
            pitch_shift = list with pitch shifts to augment training data in semi tones [-2, 1, 1, 2]
            time_stretch = list with time stretch factors to augment test data in % [0.9, 0.95, 1.05, 1.1]
            num_augment = number of augmentations, if set num will be randomly picked from pitch/time
                            otherwise do all pitch and time augmentations (default: None)
            center = If True, the signal y is padded so that frame is centered (default: False)
            minimum_duration = files will be zero padded if shorter than minimum duration in seconds
            average = return mean and std of features
            dtype = string with dtype of features (default: "float32")
            n_mfcc = number of MFCC coeffiecients to return; for example 12 or 13. Note that due to the limited number of MFCCs, the pooling action in model (e.g. ResNet) will be limited. Consider it when architecting the model.
            lifter= to emphasize the higher-order coefficients. L[n]=1+(L/2)*sin(pi*n/L). L: liftering coefficient, n: index of the MFCC coefficient. A large value of L emphasizes higher-order coefficients more strongly (default:0)
        Args:
            file (string or numpy array): file path to audio or numpy file or numpy arrays with shape [samples, channel]
            class_id (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    fftsize = extract_params.get("fftsize", FFT_SIZE_DEFAULT)
    hopsize = extract_params.get("hopsize", int(fftsize / 2))
    n_mfcc = extract_params.get("n_mfcc", 13)
    lifter = extract_params.get("lifter", 0)
    average = extract_params.get("average", False)
    dtype = extract_params.get("dtype", "float32")
    if dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    elif dtype == "float16":
        dtype = np.float16
    else:
        raise RuntimeError("Unknown dtype")

    mono = extract_params.get("mono", True)

    # for numpy we need initial sample rate
    initial_sample_rate = extract_params.get("initial_sample_rate", None)

    desired_sample_rate = extract_params.get("desired_sample_rate", None)
    # updated for better parameter name
    desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)
    normalize_audio_param = extract_params.get("normalize_audio", False)

    pitch_shift = extract_params.get("pitch_shift", None)
    time_stretch = extract_params.get("time_stretch", None)
    num_augment = extract_params.get("num_augment", None)

    center = extract_params.get("center", False)

    min_duration = extract_params.get("minimum_duration", 0)

    # support for numpy arrays with shape [samples, channel]
    if not isinstance(file, str):
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy arrays")
        y = file
        if desired_sample_rate is None:
            sr = initial_sample_rate
        else:
            y = librosa.resample(y=y, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
    elif file.endswith(".npy"):  # support for numpy files
        if initial_sample_rate is None:
            raise RuntimeError("initial_sample_rate must be set for numpy files")
        y = np.load(file)
        if desired_sample_rate is not None and initial_sample_rate != desired_sample_rate:
            y = librosa.resample(y=y, orig_sr=initial_sample_rate, target_sr=desired_sample_rate, res_type='scipy')
            sr = desired_sample_rate
        else:
            sr = initial_sample_rate
    else:  # audio
        y, sr = librosa.load(file, sr=desired_sample_rate, mono=mono, res_type='scipy')

    if normalize_audio_param is True:
        helper.normalize_audio(samples=y)

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    # pad files with zeros if they should have minimum duration
    if min_duration > 0:
        min_duration = int(min_duration * sr + 0.5)

        if y.shape[1] < min_duration:
            y_new = np.zeros((y.shape[0], min_duration), dtype=y.dtype)
            y_new[:, :y.shape[1]] = y
            y = y_new

    y_all = [y]

    # pitch shift & time stretch
    if pitch_shift is not None or time_stretch is not None:
        # do all
        if num_augment is None:
            if pitch_shift is not None:
                for p in pitch_shift:
                    y_aug = []
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.pitch_shift(y[channel], sr=sr, n_steps=p))
                    y_all.append(np.asarray(y_aug))
            if time_stretch is not None:
                for t in time_stretch:
                    y_aug = []
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.time_stretch(y[channel], rate=t))
                    y_all.append(np.asarray(y_aug))
        else:  # pick only few
            # reseeding for randomization in multiprocessing
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            for _ in range(num_augment):
                # pick pitch shift or time stretch
                if time_stretch is not None and pitch_shift is not None:
                    p = (np.random.uniform() > 0.5)
                elif time_stretch is None:
                    p = True
                else:
                    p = False

                # and select random value
                y_aug = []
                if p:
                    pitch = np.random.choice(pitch_shift, size=1)[0]
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.pitch_shift(y[channel], sr=sr, n_steps=pitch))
                else:
                    stretch = np.random.choice(time_stretch, size=1)[0]
                    for channel in range(y.shape[0]):
                        y_aug.append(librosa.effects.time_stretch(y[channel], rate=stretch))
                y_all.append(np.asarray(y_aug))

    all_x = []
    for y in y_all:
        # extract mfcc for all channels and add to "color" channel (last index)
        all_channels_x = []

        for l in range(y.shape[0]):
            x = librosa.feature.mfcc(y=y[l], sr=sr, n_fft=fftsize, hop_length=hopsize, n_mfcc=n_mfcc, lifter=lifter, center=center)
            x = x.T
            all_channels_x.append(x)
        x = np.asarray(all_channels_x)

        # move channels to last dimension (color channel)
        x = np.moveaxis(x, 0, -1)

        if x.shape[-1] == 1:
            x = np.squeeze(x)

        all_x.append(x)

    # get non augmented one
    x_orig = all_x.pop(0)

    if average:
        feat_x_mean = np.mean(x_orig, axis=0, keepdims=True)
        feat_x_std = np.std(x_orig, axis=0, keepdims=True)
        x_orig = np.hstack((feat_x_mean, feat_x_std))

    classes_orig = np.ones(x_orig.shape[0]) * class_id

    # nothing augmented
    if len(all_x) == 0:
        return [x_orig.astype(np.float32), classes_orig.astype(np.uint16), file]

    x_aug = np.vstack(all_x)
    if average:
        feat_x_mean = np.mean(x_aug, axis=0, keepdims=True)
        feat_x_std = np.std(x_aug, axis=0, keepdims=True)
        x_aug = np.hstack((feat_x_mean, feat_x_std))
    classes_aug = np.ones(x_aug.shape[0]) * class_id

    return [x_orig.astype(dtype), classes_orig.astype(np.uint16), file, x_aug.astype(dtype),
            classes_aug.astype(np.uint16)]


def extractMFCC2D(file, class_id, **extract_params):
    """Extract Mel Frequency Cepstral Coefficients (MFCC) for CNN.

        Valid parameters are:
            fftsize = fftsize for STFT transformation (default: 2048)
            hopsize = hopsize for STFT transformation (default: winsize/2)
            mono: if true channels will be converted to mono (default: True)
            desired_sample_rate = if set change sample rate of audio (default: None)
            patchsize = number of time frames per CNN patch
            patchsize_in_seconds = set CNN patch size in seconds. Mutually exclusive use with 'patchsize' (default: None - patchsize in time frames used)
            patchhop = hppsize for creating patches (default: patchsize - no overlap)
            patchhop_in_seconds = set overlaping of patches in seconds. Mutually exclusive use with 'patchhop' (default: patchsize_in_seconds)
            pitch_shift = list with pitch shifts to augment training data in semi tones [-2, 1, 1, 2]
            time_stretch = list with time stretch factors to augment test data in % [0.9, 0.95, 1.05, 1.1]
            num_augment = number of augmentations, if set num will be randomly picked from pitch/time
                            otherwise do all pitch and time augmentations (default: None)
            minimum_duration = files will be zero padded if shorter than minimum duration in seconds
            center = If True, the signal y is padded so that frame is centered (default: False)
            dtype = string with dtype of features (default: "float32")
        Args:
            file (string): string of file path
            class_id (dict): class id of file
            extract_params (**kwargs): dict with additional parameters (alf interface)
        Returns:
            [x, y, file name]
        """
    dtype = extract_params.get("dtype", "float32")
    if dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    elif dtype == "float16":
        dtype = np.float16
    else:
        raise RuntimeError("Unknown dtype")

    patchsize = extract_params.get("patchsize", None)
    patchsize_in_seconds = extract_params.get("patchsize_in_seconds", None)

    if patchsize_in_seconds is not None and patchsize is not None:
        LOGGER.error(
            "Use either 'patchsize' or 'patchsize_in_seconds' when defining the feature extraction function in config file")
        raise RuntimeError

    res = extractMFCC(file, class_id, **extract_params)

    if patchsize_in_seconds is not None:
        # for conversion (bit complicated since this information is normally extracted with STFT
        fftsize = extract_params.get("fftsize", FFT_SIZE_DEFAULT)
        hopsize = extract_params.get("hopsize", int(fftsize / 2))
        desired_sample_rate = extract_params.get("desired_sample_rate", None)

        # updated for better parameter name
        desired_sample_rate = extract_params.get("sample_rate", desired_sample_rate)

        if desired_sample_rate is None:
            LOGGER.error("desired_sample_rate must be set when patch size is set in seconds")
            raise ValueError("desired_sample_rate must be set when patch size is set in seconds")

        patchhop_in_seconds = extract_params.get("patchhop_in_seconds", patchsize_in_seconds)

        # librosa stft is centered (and padded) so full file can be used
        patchsize = int((patchsize_in_seconds * desired_sample_rate / hopsize) + 1)
        patchhop = int((patchhop_in_seconds * desired_sample_rate / hopsize) + 1)
    else:
        # take full patch
        if patchsize is None:
            patchsize = res[0].shape[0]

        patchhop = extract_params.get("patchhop", patchsize)

    patches = extraction_tools.extract_2D_patches_from_features(res[0], patchsize, patchhop)

    classes = np.ones(patches.shape[0]) * class_id

    # not augmented
    if len(res) < 5:
        return [patches.astype(res[0].dtype), classes.astype(np.uint16), file]

    # take patches from full augmented (overlapt between files shoulnd't hurt since it's the same class)
    patches_aug = extraction_tools.extract_2D_patches_from_features(res[3], patchsize, patchhop)
    classes_aug = np.ones(patches_aug.shape[0]) * class_id
    return [patches.astype(res[0].dtype), classes.astype(np.uint16), file, patches_aug.astype(dtype),
            classes_aug.astype(np.uint16)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import librosa.display

    file = librosa.ex('trumpet')

    extract_params = dict()
    extract_params["fftsize"] = 2048
    extract_params["hopsize"] = 1024
    extract_params["mel_bands"] = 64
    extract_params["n_mfcc"] = 13
    extract_params["lifter"] = 0
    extract_params["power"] = 2
    extract_params["log_spec"] = True
    extract_params["pcen"] = False
    extract_params["mono"] = True

    _, sr = librosa.load(file, sr=None, mono=True, res_type='scipy')

    # test methods and plot
    x_orig, _, _ = extractSTFT(file, 0, **extract_params)

    fig, ax = plt.subplots(nrows=5, sharex=True)
    librosa.display.specshow(x_orig.T, sr=sr, x_axis='time', y_axis='mel', ax=ax[0],
                             hop_length=extract_params["hopsize"])
    ax[0].set(title='log mel spec')
    ax[0].label_outer()

    # pcen
    extract_params["pcen"] = True
    x, _, _ = extractSTFT(file, 0, **extract_params)
    librosa.display.specshow(x.T, sr=sr, x_axis='time', y_axis='mel', ax=ax[1], hop_length=extract_params["hopsize"])
    ax[1].set(title='Per-channel energy normalization')
    ax[1].label_outer()

    # magn spec
    extract_params["pcen"] = False
    extract_params["mel_bands"] = 0
    extract_params["power"] = 1
    x, _, _ = extractSTFT(file, 0, **extract_params)
    librosa.display.specshow(x.T, sr=sr, x_axis='time', y_axis='linear', ax=ax[2], hop_length=extract_params["hopsize"])
    ax[2].set(title='magn spec')
    ax[2].label_outer()

    # quick test
    extract_params["patchsize"] = 50
    x, _, _ = extractSTFT2D(file, 0, **extract_params)

    # cqt
    extract_params = dict()
    extract_params["hopsize"] = 1024
    extract_params["bins_per_octave"] = 12
    extract_params["n_octaves"] = 7
    extract_params["fmin"] = 80
    x, _, _ = extractCQT(file, 0, **extract_params)
    librosa.display.specshow(x.T, sr=sr, x_axis='time', y_axis='cqt_hz', ax=ax[3], hop_length=extract_params["hopsize"],
                             fmin=extract_params["fmin"], bins_per_octave=extract_params["bins_per_octave"])
    ax[3].set(title='cqt')
    ax[3].label_outer()

    # quick test
    extract_params["patchsize"] = 50
    x, _, _ = extractCQT2D(librosa.ex('trumpet'), 0, **extract_params)

    # mfcc
    x, _, _ = extractMFCC(file, 0, **extract_params)
    librosa.display.specshow(x.T, sr=sr, x_axis='time', ax=ax[4], hop_length=extract_params["hopsize"])
    ax[4].set(title='MFCC feature')
    ax[4].label_outer()

    plt.show()

    # test time & pitch
    extract_params = dict()
    extract_params["fftsize"] = 2048
    extract_params["hopsize"] = 1024
    extract_params["mel_bands"] = 64
    extract_params["power"] = 2
    extract_params["log_spec"] = True
    extract_params["pcen"] = False
    extract_params["mono"] = True

    extract_params["pitch_shift"] = [-2, 2]
    extract_params["time_stretch"] = [0.9, 1.1]

    x, y, _, x_aug1, y_aug1 = extractSTFT(file, 0, **extract_params)
    x, y, _, x_aug1, y_aug1 = extractMFCC(file, 0, **extract_params)

    extract_params["num_augment"] = 2
    x, y, _, x_aug2, y_aug2 = extractSTFT(file, 0, **extract_params)
    x, y, _, x_aug2, y_aug2 = extractMFCC(file, 0, **extract_params)

    extract_params["time_stretch"] = None
    extract_params["num_augment"] = None
    x, y, _, x_aug3, y_aug3 = extractSTFT(file, 0, **extract_params)
    x, y, _, x_aug3, y_aug3 = extractMFCC(file, 0, **extract_params)

    extract_params["time_stretch"] = [0.9, 1.1]
    extract_params["pitch_shift"] = None
    x, y, _, x_aug4, y_aug4 = extractSTFT(file, 0, **extract_params)
    x, y, _, x_aug4, y_aug4 = extractMFCC(file, 0, **extract_params)

    print("Finished")
