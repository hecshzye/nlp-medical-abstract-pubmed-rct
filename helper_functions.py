# functions for future use

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to import and resize images
def load_prepare_images(filename, img_shape=224, scale=True):
    """[summary]

    Args:
        filename ([type]): [description]
        img_shape (int, optional): [description]. Defaults to 224.
        scale (bool, optional): [description]. Defaults to True.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(image)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255.
    else:
        return img


# Function to plot a confusion matrix
import numpy as np 
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
def create_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """[summary]

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        classes ([type], optional): [description]. Defaults to None.
        figsize ([type], optional): [description]. Defaults to (10, 10).
        text_size (int, optional): [description]. Defaults to 15.
        norm (bool, optional): [description]. Defaults to False.
        savefig (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Plot confusion matrix 
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    if classes: 
        labels = classes
    else:
        labels = np.arange(cm.shap[0])
    ax.set(title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label",
           xticks = np.arange(n_classes),
           yticks = np.arange(n_classes),
           xticklabels = labels,
           yticklabels = labels)   

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, k]*100:.1f}%",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

        else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)

    if savefig:
        fig.savefig("confusion_matrix.png")


# Function to predict images and plot
def pred_plot(model, filename, class_names):
    img = load_prepare_images(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
        
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);

import datetime

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saved Tensorboard logs to: {log_dir}")  
    return tensorboard_callback

# Function to plot validation and training separately
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss]"]))

    plt.plot(epochs, loss, label="Training_loss")
    plt.plot(epochs, val_loss, label="Validation_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label="Training_accuracy")
    plt.plot(epochs, val_accuracy, label="Validation_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();

def compare_history(original_history, new_history, inital_epochs=5):
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training_accuracy")
    plt.plot(total_val_acc, label="Validation_accuracy")
    plt.plot([initial_epochs-1, initial_epoch-1],
              plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training_loss")
    plt.plot(total_val_loss, label="Validation_loss")
    plt.plot([initial_epochs-1, initial_epoch-1],
              plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

# Function to unzip a file
import zipfile

def unzip_data(filename):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()


# Function to walkthrough a directory and return a list of files
import os

def walk_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"Directories {len(dirnames)} and images {len(filenames)} in '{dirpath}'. ")


# Function for evaluation, accuraccy, precision_recall_f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results                 
    