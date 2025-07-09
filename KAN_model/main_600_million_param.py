import argparse
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_final import load_data_zip, load_data, compute_melspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from kolmogorov_arnold_networks import KAN  # Assuming a KAN implementation is available

def create_kan_model(input_shape, num_classes):
    """Creates a simplified Kolmogorov–Arnold Network (KAN) for classification."""
    inp = tf.keras.layers.Input(shape=input_shape)

    # Feature extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    # KAN block
    x = tf.keras.layers.BatchNormalization()(x)
    x = KAN(output_dim=256)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Classification head
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    return model



def plot_accuracy_loss(history, output_dir):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Calculate percentages

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, cmap='Blues', fmt='.2%', xticklabels=classes, yticklabels=classes, annot_kws={"size": 16})  # Adjust font size
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_f1_scores(y_true, y_pred, classes, output_dir):
    f1 = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=classes, y=f1)
    plt.title('F1 Scores for Each Class')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'f1_scores.png'))
    plt.close()

def compute_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def compute_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    print(f'Precision: {precision:.4f}')
    return precision

def compute_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f'Recall: {recall:.4f}')
    return recall

def compute_f1_score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 Score: {f1:.4f}')
    return f1

def save_metrics(metrics, output_dir):
    metrics_path = f'{output_dir}/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Metrics saved to {metrics_path}')

def save_model_summary(model, output_dir):
    # Capture the model summary as a string
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Save the model summary to a JSON file
    model_summary_path = os.path.join(output_dir, 'model_summary.json')
    with open(model_summary_path, 'w') as f:
        json.dump({"model_summary": model_summary}, f, indent=4)
    print(f'Model summary saved to {model_summary_path}')


def save_training_history(history, output_dir):
    # Extract the history dictionary
    history_dict = history.history
    
    # Save the history to a JSON file
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f'Training history saved to {history_path}')

def save_model_checkpoint(model, output_dir):
    # Create a ModelCheckpoint callback to save the model during training
    checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,  # Save only the best model
        monitor='val_loss',  # Monitor validation loss to save best model
        mode='min',  # Save the model with the minimum validation loss
        verbose=1
    )
    return checkpoint_callback


def main_func(train_dir, test_dir, batch_size, epochs, output_dir, plot_confusion_matrix_flag, plot_accuracy_flag, plot_f1_scores_flag):
    # Load data
    if train_dir.endswith('.zip') or train_dir.endswith('.7z'):
        X_train, y_train, X_test, y_test, num_classes, classes = load_data_zip(train_dir, test_dir, output_dir)
    else:
        X_train, y_train, X_test, y_test, num_classes, classes = load_data(train_dir, test_dir, output_dir)

    input_shape = X_train.shape[1:]

    # Build model
    model = create_kan_model(input_shape, num_classes)
    model.summary()
    save_model_summary(model, output_dir)

    checkpoint_callback = save_model_checkpoint(model, output_dir)

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    save_training_history(history, output_dir)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')

    # Optionally plot accuracy and loss curves
    if plot_accuracy_flag:
        plot_accuracy_loss(history, output_dir)

    # Predict on test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute evaluation metrics
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1_score": compute_f1_score(y_true, y_pred)
    }

    # Save metrics to a file
    save_metrics(metrics, output_dir)

    # Optionally plot confusion matrix
    if plot_confusion_matrix_flag:
        plot_confusion_matrix(y_true, y_pred, classes, output_dir)

    # Optionally plot F1 scores
    if plot_f1_scores_flag:
        plot_f1_scores(y_true, y_pred, classes, output_dir)
