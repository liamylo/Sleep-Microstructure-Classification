import numpy as np
import seaborn as sns

from sklearn import metrics
from matplotlib import pyplot as plt


def show_basic_dataframe_info(dataframe):
    print('\nNumber of columns in the dataframe: ', dataframe.shape[1])
    print('Number of rows in the dataframe:    ',  dataframe.shape[0])
    print('\n', dataframe.head(5))


def plot_labels(labels, title):
    labels.value_counts().plot(kind='bar',
                               title=title)
    plt.show()


def plot_event(event, data):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4,
                                             figsize=(15, 10),
                                             sharex=True)
    plot_axis(ax0, data['Datetime'], data['F2-F4[uV]'], 'F2-F4[uV]')
    plot_axis(ax1, data['Datetime'], data['F4-C4[uV]'], 'F4-C4[uV]')
    plot_axis(ax2, data['Datetime'], data['C4-A1[uV]'], 'C4-A1[uV]')
    plot_axis(ax3, data['Datetime'], data['ECG1-ECG2[uV]'], 'ECG1-ECG2[uV]')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(event)
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_data(dataset):

    plt.plot(dataset['Datetime'], dataset['Event'], dataset['F2-F4[uV]'], 'b.')
    plt.show()
    plt.plot(dataset['Datetime'], dataset['Event'], dataset['F4-C4[uV]'], 'y.')
    plt.show()
    plt.plot(dataset['Datetime'], dataset['Event'], dataset['C4-A1[uV]'], 'r.')
    plt.show()
    plt.plot(dataset['Datetime'], dataset['Event'], dataset['ECG1-ECG2[uV]'], 'g.')
    plt.show()


def show_model_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()


def show_confusion_matrix(validations, predictions, labels, title):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
