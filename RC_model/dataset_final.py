import numpy as np
import os
import glob
import librosa
import shutil
import zipfile
import py7zr
import tensorflow as tf
from tqdm import tqdm

def compute_melspec(fn_wav, n_bins=128, fixed_length=300):
    x, fs = librosa.load(fn_wav, mono=True, sr=44100)
    S = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_bins, fmax=fs/2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if S_dB.shape[1] < fixed_length:
        pad_width = fixed_length - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :fixed_length]
    
    return S_dB, fs, len(x) / fs

def process_directory(directory):
    print(f"Processing directory: {directory}")
    fn_wav_list = []
    class_label = []

    countries = glob.glob(os.path.join(directory, '*'))
    for country in countries:
        country_name = os.path.basename(country)
        print(f"  Country: {country_name}")
        current_fn_wav_list = sorted(glob.glob(os.path.join(country, '*.wav')))
        for fn_wav in current_fn_wav_list:
            fn_wav_list.append(fn_wav.replace("\\", "/"))
            class_label.append(country_name)

    return fn_wav_list, class_label

def print_class_distribution(class_label, set_name):
    unique, counts = np.unique(class_label, return_counts=True)
    print(f"Class distribution for {set_name}:")
    for cls, count in zip(unique, counts):
        print(f"  Class '{cls}': {count} files")

def save_class_distribution_to_file(class_label, set_name, output_dir):
    unique, counts = np.unique(class_label, return_counts=True)
    with open(os.path.join(output_dir, f'{set_name}_class_distribution.txt'), 'w') as f:
        f.write(f"Class distribution for {set_name}:\n")
        for cls, count in zip(unique, counts):
            f.write(f"  Class '{cls}': {count} files\n")

def mixup_data(x1, x2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2

def extract_and_process(directory, temp_dir):
    if directory.endswith('.zip'):
        with zipfile.ZipFile(directory, 'r') as archive:
            archive.extractall(path=temp_dir)
    elif directory.endswith('.7z'):
        with py7zr.SevenZipFile(directory, mode='r') as archive:
            archive.extractall(path=temp_dir)
    return process_directory(temp_dir)

def balance_and_mixup(data, labels, class_labels):
    unique_classes, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)

    if not np.all(counts == max_count):
        balanced_data = []
        balanced_labels = []
        new_data = []

        for cls in tqdm(unique_classes, desc="Balancing classes"):
            cls_indices = np.where(labels == cls)[0]
            cls_data = data[cls_indices]
            num_to_add = max_count - len(cls_data)

            balanced_data.extend(cls_data)
            balanced_labels.extend([cls] * len(cls_data))

            if num_to_add > 0:
                while num_to_add > 0:
                    if len(cls_data) >= 2:
                        idx1, idx2 = np.random.choice(len(cls_data), 2, replace=False)
                        new_sample = mixup_data(cls_data[idx1], cls_data[idx2])
                        balanced_data.append(new_sample)
                        balanced_labels.append(cls)
                        new_data.append(new_sample)
                        num_to_add -= 1
                    else:
                        break

        unique_balanced_classes, balanced_counts = np.unique(balanced_labels, return_counts=True)
        print("\nClass distribution after balancing:")
        for cls, count in zip(unique_balanced_classes, balanced_counts):
            print(f"  Class '{class_labels[cls]}': {count} files")

        return np.array(balanced_data), np.array(balanced_labels), np.array(new_data)

    return data, labels, np.array([])

def load_data_zip(train_dir, test_dir, output_dir):
    train_temp_dir = 'train_temp'
    test_temp_dir = 'test_temp'
    os.makedirs(train_temp_dir, exist_ok=True)
    os.makedirs(test_temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_fn_wav_list, train_class_label = extract_and_process(train_dir, train_temp_dir)
    test_fn_wav_list, test_class_label = extract_and_process(test_dir, test_temp_dir)

    avg_sample_rate = 0.0
    avg_duration = 0.0

    train_output_dir = os.path.join(output_dir, 'Train')
    os.makedirs(train_output_dir, exist_ok=True)

    for fn_wav, country in zip(train_fn_wav_list, train_class_label):
        country_dir = os.path.join(train_output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        shutil.copy(fn_wav, country_dir)

    print(f"Total number of training samples: {len(train_fn_wav_list)}")
    print_class_distribution(train_class_label, "training set")
    save_class_distribution_to_file(train_class_label, "training set", output_dir)

    test_output_dir = os.path.join(output_dir, 'Test')
    os.makedirs(test_output_dir, exist_ok=True)

    for fn_wav, country in zip(test_fn_wav_list, test_class_label):
        country_dir = os.path.join(test_output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        shutil.copy(fn_wav, country_dir)

    print(f"Total number of test samples: {len(test_fn_wav_list)}")
    print_class_distribution(test_class_label, "test set")
    save_class_distribution_to_file(test_class_label, "test set", output_dir)

    unique_classes = sorted(list(set(train_class_label + test_class_label)))

    class_id_train = np.array([unique_classes.index(cls) for cls in train_class_label])
    class_id_test = np.array([unique_classes.index(cls) for cls in test_class_label])

    train_feat = [compute_melspec(fn_wav)[0] for fn_wav in tqdm(train_fn_wav_list, desc="Computing mel spectrograms for training set")]
    test_feat = [compute_melspec(fn_wav)[0] for fn_wav in tqdm(test_fn_wav_list, desc="Computing mel spectrograms for test set")]

    train_feat = np.array(train_feat)
    test_feat = np.array(test_feat)

    train_feat, class_id_train, new_train_data = balance_and_mixup(train_feat, class_id_train, unique_classes)
    test_feat, class_id_test, new_test_data = balance_and_mixup(test_feat, class_id_test, unique_classes)

    X_train = train_feat[:, :, :, np.newaxis]
    X_test = test_feat[:, :, :, np.newaxis]

    y_train = tf.keras.utils.to_categorical(class_id_train, num_classes=len(unique_classes))
    y_test = tf.keras.utils.to_categorical(class_id_test, num_classes=len(unique_classes))

    X_train -= np.mean(X_train)
    X_train /= np.std(X_train)
    X_test -= np.mean(X_test)
    X_test /= np.std(X_test)

    print("\nLogging information after balancing:")

    # Logging after balancing
    print_class_distribution(class_id_train, "balanced training set")
    save_class_distribution_to_file(class_id_train, "balanced_training_set", output_dir)

    print_class_distribution(class_id_test, "balanced test set")
    save_class_distribution_to_file(class_id_test, "balanced_test_set", output_dir)

    return X_train, y_train, X_test, y_test, len(unique_classes), unique_classes

def load_data(train_dir, test_dir, output_dir):
    train_fn_wav_list, train_class_label = process_directory(train_dir)
    test_fn_wav_list, test_class_label = process_directory(test_dir)

    avg_sample_rate = 0.0
    avg_duration = 0.0

    train_output_dir = os.path.join(output_dir, 'Train')
    os.makedirs(train_output_dir, exist_ok=True)

    for fn_wav, country in zip(train_fn_wav_list, train_class_label):
        country_dir = os.path.join(train_output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        #shutil.copy(fn_wav, country_dir)

    print(f"Total number of training samples: {len(train_fn_wav_list)}")
    print_class_distribution(train_class_label, "training set")
    save_class_distribution_to_file(train_class_label, "training set", output_dir)

    shutil.rmtree(train_output_dir)

    test_output_dir = os.path.join(output_dir, 'Test')
    os.makedirs(test_output_dir, exist_ok=True)

    for fn_wav, country in zip(test_fn_wav_list, test_class_label):
        country_dir = os.path.join(test_output_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        #shutil.copy(fn_wav, country_dir)

    print(f"Total number of test samples: {len(test_fn_wav_list)}")
    print_class_distribution(test_class_label, "test set")
    save_class_distribution_to_file(test_class_label, "test set", output_dir)

    shutil.rmtree(test_output_dir)

    unique_classes = sorted(list(set(train_class_label + test_class_label)))

    class_id_train = np.array([unique_classes.index(cls) for cls in train_class_label])
    class_id_test = np.array([unique_classes.index(cls) for cls in test_class_label])

    train_feat = [compute_melspec(fn_wav)[0] for fn_wav in tqdm(train_fn_wav_list, desc="Computing mel spectrograms for training set")]
    test_feat = [compute_melspec(fn_wav)[0] for fn_wav in tqdm(test_fn_wav_list, desc="Computing mel spectrograms for test set")]

    train_feat = np.array(train_feat)
    test_feat = np.array(test_feat)

    #train_feat, class_id_train, new_train_data = balance_and_mixup(train_feat, class_id_train, unique_classes)
    #test_feat, class_id_test, new_test_data = balance_and_mixup(test_feat, class_id_test, unique_classes)

    X_train = train_feat[:, :, :, np.newaxis]
    X_test = test_feat[:, :, :, np.newaxis]

    y_train = tf.keras.utils.to_categorical(class_id_train, num_classes=len(unique_classes))
    y_test = tf.keras.utils.to_categorical(class_id_test, num_classes=len(unique_classes))

    X_train -= np.mean(X_train)
    X_train /= np.std(X_train)
    X_test -= np.mean(X_test)
    X_test /= np.std(X_test)

    print("\nLogging information after balancing:")

    # Logging after balancing
    print_class_distribution(class_id_train, "balanced training set")
    save_class_distribution_to_file(class_id_train, "balanced_training_set", output_dir)

    print_class_distribution(class_id_test, "balanced test set")
    save_class_distribution_to_file(class_id_test, "balanced_test_set", output_dir)

    return X_train, y_train, X_test, y_test, len(unique_classes), unique_classes

