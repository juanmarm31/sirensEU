import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_data, load_data_zip
from sklearn.metrics import accuracy_score, confusion_matrix
from kolmogorov_arnold_network import KAN  # Hypothetical KAN library


"""
Project siren classification
ESA 1 - Sound Event Detection - 1
Juan Manuel Rodriguez Mejia 
juan-manuel.rodriguez-mejia@tu-ilmenau.de

Last update: 29.05.2024

TODO

implement generator (to be used in the next notebook for data augmentation with audiomentations library)
implement CNN (FSD50k) and CRNN basic
Outline

In this notebook, we revise the M1 notebook and use a small dataset of animal sounds extracted from the ESC50 dataset. We will study how to

use the audiomentations Python library for data augmentation and how
to implement a custom generator for our training to apply the data augmentation during training.
"""

"""
conda create --name SiCaPKF
conda activate SiCaPKF

pip install wget
pip install audiomentations
"""

def plot_confusion_matrix_from_matrix(confusion_matrix, classIDDict, meta_info, dir_save, title="Confusion matrix from matrix", formats=None):
    """
    Plots a confusion matrix from a given matrix.

    :param confusion_matrix: (ndarray) Confusion Matrix as numpy array.
    :param classIDDict: (dict) Dictionary of all classes with ID and name.
    :param meta_info: (string) Unique string for labeling the file.
    :param dir_save: (string) Output path for saving the plots.
    :param title: (string) Title for the plot (above the chart).
    :param formats: (list) List of formats to save figs (e.g., ["pdf", "png"]).
    """
    classIDs = list(classIDDict.values())
    labels = list(classIDDict.keys())
    fontsize = int(36 / confusion_matrix.shape[0])
    if fontsize < 2:
        fontsize = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tick_params(labelsize=fontsize)

    fmt = '.2f'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    ax.set_aspect('equal')
    fig.tight_layout()

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    if formats is None:
        plt.savefig(os.path.join(dir_save, meta_info + 'confusion_matrix.png'), dpi=600)
    else:
        if isinstance(formats, list):
            for fm in formats:
                plt.savefig(os.path.join(dir_save, meta_info + f'confusion_matrix.{fm}'), format=fm, dpi=300)
        elif isinstance(formats, str):
            plt.savefig(os.path.join(dir_save, meta_info + f'confusion_matrix.{formats}'), format=formats, dpi=300)
    
    np.save(os.path.join(dir_save, meta_info + 'confusion_matrix.npy'), confusion_matrix)
    plt.show()
    plt.clf()
    plt.close()

def plot_confusion_matrix_from_prediction(class_true, class_pred, classIDDict, file_dir, normalize=True, title=None, meta_info=""):
    """
    This function computes and plots the confusion matrix from predicted values.

    :param class_true: (array) Ground truth labels.
    :param class_pred: (array) Predicted labels.
    :param classIDDict: (dict) Dictionary of all classes with ID and name.
    :param file_dir: (string) Output directory to save the plots.
    :param normalize: (bool) If True, normalize the confusion matrix.
    :param title: (string) Title for the plot.
    :param meta_info: (string) Unique string for labeling the file.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(class_true, class_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix_from_matrix(confusion_matrix=cm, classIDDict=classIDDict, meta_info=meta_info, dir_save=file_dir, title=title)

def plot_confusion_matrix(y_true, y_pred, classes, unique_classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
    
    ax.set_aspect('equal')  # Ensure equal aspect ratio for x and y axes
    plt.xlabel("Predicted Label")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()  # Show the plot for visual confirmation

def plot_accuracy_loss(history, output_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    #plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    #plt.show()


def create_kan_model(input_shape, num_classes):
    model = KAN(input_shape=input_shape, num_classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main_func(train_dir, test_dir, batch_size, epochs, output_dir, plot_confusion_matrix_flag, plot_accuracy_flag):
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError('Train or test directory does not exist.')
    
    if (train_dir.endswith(".zip") and test_dir.endswith(".zip")) or (train_dir.endswith(".7z") and test_dir.endswith(".7z")):
        X_train, y_train, X_test, y_test, num_classes, unique_classes = load_data_zip(train_dir, test_dir)
    else:
        X_train, y_train, X_test, y_test, num_classes, unique_classes = load_data(train_dir, test_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    model = create_kan_model(X_train.shape[1:], num_classes)
    model.summary()
    
    checkpoint_path = os.path.join(output_dir, 'checkpoint.keras')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max', verbose=1)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint_callback])
    
    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)
    
    if plot_accuracy_flag:
        plot_accuracy_loss(history, output_dir)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    classIDDict = {i: unique_classes[i] for i in range(num_classes)}
    if plot_confusion_matrix_flag:
        plot_confusion_matrix(y_true_classes, y_pred_classes, classIDDict, unique_classes, output_dir)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f'Test Accuracy: {accuracy:.4f}')
