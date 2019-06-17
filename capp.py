from __future__ import print_function

from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn.model_selection as sk
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.python.keras.utils import np_utils

import dataset, models, plots, signals
import labels as labels_script


def setup():
    # Set up some standard parameters
    pd.options.display.float_format = '{:.1f}'.format
    sns.set()  # Default seaborn look and feel.
    plt.style.use('ggplot')

    print('Keras version: ', tf.keras.__version__)


def describe(data_set):
    # Describe the data
    plots.show_basic_dataframe_info(data_set)
    plots.plot_labels(data_set['Event'], title='Sleeping Examples by Event Type')
    # plots.plot_data(data_set)
    '''
    for event in np.unique(df['Event']):
        subset = df[df['Event'] == event][:180]
        plot_event(event, subset)
    '''


def smote(data, labels):

    sm = SMOTE()
    x_train, y_train = sm.fit_sample(data, labels.ravel())

    print("\nCounts of label '0' before and after oversampling: {} -> {}".format(sum(labels == 0), sum(y_train == 0)))
    print("Counts of label '1' before and after oversampling: {} -> {}".format(sum(labels == 1), sum(y_train == 1)))
    print("Dataset size increased from {} to {}".format(sum(labels == 1) + sum(labels == 0), sum(y_train == 1) + sum(y_train == 0)))

    return x_train, y_train


# Define column name of the label vector
LABEL = 'EventEncoded'


def convert_string_to_integer(data_set):
    # Transform the labels from String to Integer via LabelEncoder
    label_encoder = preprocessing.LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    data_set[LABEL] = label_encoder.fit_transform(data_set['Event'].values.ravel())
    # data_set['Event'] = label_encoder.fit_transform(data_set['Event'].values.ravel())

    # TODO Define the number of classes
    # -> Number of classes: the amount of nodes for our output layer in the neural network.
    # Since we want our neural network to predict the type of activity,
    # we will take the number of classes from the encoder that we have used earlier.
    num_classes = label_encoder.classes_.size
    # print('\n Number of classes: ', num_classes)
    # print(' Available classes: ', list(label_encoder.classes_))

    return data_set, num_classes


def initialize_model(TIME_PERIODS, input_shape, num_sensors, num_classes, x_train, y_train_hot, x_test, y_train, y_test):

    model = models.cnn(time_periods=TIME_PERIODS,
                       input_shape=input_shape,
                       number_of_sensors=num_sensors,
                       number_of_classes=num_classes,
                       optimizer='adam',
                       filters=100,
                       kernel_size=3)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../best_models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]

    # TODO Train the model
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model.fit(x_train,
                        y_train_hot,
                        batch_size=80,
                        epochs=50,
                        callbacks=callbacks_list,
                        validation_split=0.2,
                        verbose=1)

    plots.show_model_history(history)

    # plot metrics
    plt.plot(history.history['acc'])
    plt.show()

    plt.plot(history.history['mean_squared_error'])
    plt.show()

    y_pred_train, max_y_pred_train = train_model(model, x_train, y_train)
    y_pred_test, max_y_pred_test = test_model(model, x_test, y_test)

    return y_pred_train, max_y_pred_train, y_pred_test, max_y_pred_test


def train_model(model, x_train, y_train):
    # TODO Test the model on the training data
    y_pred_train = model.predict(x_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)

    # TODO Show confusion matrices and classification reports.
    # Print confusion matrix for training data
    plots.show_confusion_matrix(y_train, max_y_pred_train, LABELS,
                                title='Confusion Matrix for Training Data')
    print_classification_report(y_train, max_y_pred_train)

    return y_pred_train, max_y_pred_train


def test_model(model, x_test, y_test):
    # TODO Test the model on the testing data
    y_pred_test = model.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)

    # Print confusion matrix for testing data
    plots.show_confusion_matrix(y_test, max_y_pred_test, LABELS,
                                title='Confusion Matrix for Testing Data')

    print_classification_report(y_test, max_y_pred_test)

    return y_pred_test, max_y_pred_test


LABELS = ['CAP',
          'NON-CAP'
          ]


def evaluate(t, predict, criterion):
    """
    > Είσοδος t : διάνυσμα με τους πραγματικούς στόχους (0/1)
    > Είσοδος predict : διάνυσμα με τους εκτιμώμενους στόχους (0/1)
    > Είσοδος criterion : text-string με τις εξής πιθανές τιμές:
         'accuracy', 'precision', 'recall', 'fmeasure', 'sensitivity', 'specificity'
    """
    # Ypologismos true positive, true negative, false positive, false negative
    tp = float(sum((predict == 1) & (t == 1)))
    tn = float(sum((predict == 0) & (t == 0)))
    fp = float(sum((predict == 1) & (t == 0)))
    fn = float(sum((predict == 0) & (t == 1)))

    try:
        # Ypologismos twn pithanwn timwn gia to criterion
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        fmeasure = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        fmeasure = 0
    try:
        sensitivity = tp / (tp + fn)
    except ZeroDivisionError:
        sensitivity = 0
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0

    # Anathesh timhs gia kathe criterion
    value = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

    # Έξοδος value : η τιμή του κριτηρίου που επιλέξαμε.
    return value[criterion]


def print_classification_report(y, max_y_pred):
    print('_________________________________________________________________\n',
          'Classification Report for Training Data:\n',
          classification_report(y, max_y_pred),
          '\nTrain STD: ', max_y_pred.std(),
          '\nAccuracy: ', accuracy_score(y, max_y_pred),
          '\nF1 score: ', f1_score(y, max_y_pred))


def cross_validation(folds, data_set, num_classes):
    data_set.drop(columns=['Event'], axis=1, inplace=True)

    test_max_accuracy = \
        test_sum_accuracy = test_sum_precision = test_sum_recall = \
        test_sum_fmeasure = test_sum_sensitivity = test_sum_specificity = 0

    train_max_accuracy = \
        train_sum_accuracy = train_sum_precision = train_sum_recall = \
        train_sum_fmeasure = train_sum_sensitivity = train_sum_specificity = 0

    for fold in range(folds):

        # TODO Split train and test sets
        df_train, df_test = sk.train_test_split(data_set, test_size=0.2, shuffle=False)

        # TODO Smote train set and make data set even
        x_train, y_train = smote(data=df_train.iloc[:, :-1],
                                 labels=df_train[LABEL])

        new_df_train = pd.DataFrame(np.column_stack([x_train, y_train]), columns=list(df_train))

        # The number of steps within one time segment, 1 second has 100 steps but we want to take 2 second intervals
        # therefore in our case we have to take 200 steps.
        TIME_PERIODS = 200

        # The steps to take from one segment to the next;
        # if this value is equal to TIME_PERIODS, then there is no overlap between the segments.
        # In our case, we want to have 1 second overlap, therefore the step distance will be 100.
        STEP_DISTANCE = int(TIME_PERIODS / 2)

        x_train, y_train = dataset.create_segments_and_labels(new_df_train, TIME_PERIODS, STEP_DISTANCE, LABEL)
        x_test, y_test = dataset.create_segments_and_labels(df_test, TIME_PERIODS, STEP_DISTANCE, LABEL)

        # TODO Define input and output dimensions
        # -> Number of time periods: the number of time periods within one record.
        # -> Number of sensors: the number of sensors we used from one record.
        num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
        input_shape = num_time_periods * num_sensors

        # TODO Reshape the train and test sets accordingly
        x_train = x_train.reshape(x_train.shape[0], input_shape)
        x_test = x_test.reshape(x_test.shape[0], input_shape)

        # TODO Convert the feature and label data into float so that it will be accepted by Keras
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')

        # TODO One hot encode the labels -> MUST EXECUTE ONLY ONCE!
        y_train_hot = np_utils.to_categorical(y_train, num_classes)

        y_pred_train, max_y_pred_train, y_pred_test, max_y_pred_test = initialize_model(TIME_PERIODS, input_shape, num_sensors, num_classes,
                                                        x_train, y_train_hot, x_test, y_train, y_test)

        # Call function evaluate for all criterion
        accuracy = evaluate(y_train, max_y_pred_train, "accuracy")
        precision = evaluate(y_train, max_y_pred_train, "precision")
        recall = evaluate(y_train, max_y_pred_train, "recall")
        fmeasure = evaluate(y_train, max_y_pred_train, "fmeasure")
        sensitivity = evaluate(y_train, max_y_pred_train, "sensitivity")
        specificity = evaluate(y_train, max_y_pred_train, "specificity")

        if accuracy > train_max_accuracy:
            train_max_accuracy = accuracy

        # Calculate results
        train_sum_accuracy += accuracy
        train_sum_precision += precision
        train_sum_recall += recall
        train_sum_fmeasure += fmeasure
        train_sum_sensitivity += sensitivity
        train_sum_specificity += specificity

    # Call function evaluate for all criterion
        accuracy = evaluate(y_test, max_y_pred_test, "accuracy")
        precision = evaluate(y_test, max_y_pred_test, "precision")
        recall = evaluate(y_test, max_y_pred_test, "recall")
        fmeasure = evaluate(y_test, max_y_pred_test, "fmeasure")
        sensitivity = evaluate(y_test, max_y_pred_test, "sensitivity")
        specificity = evaluate(y_test, max_y_pred_test, "specificity")

        if accuracy > test_max_accuracy:
            test_max_accuracy = accuracy

        # Calculate results
        test_sum_accuracy += accuracy
        test_sum_precision += precision
        test_sum_recall += recall
        test_sum_fmeasure += fmeasure
        test_sum_sensitivity += sensitivity
        test_sum_specificity += specificity

    print("\n\nAccuracy = %f" % (train_sum_accuracy / folds))
    print("Precision = %f" % (train_sum_precision / folds))
    print("Recall = %f" % (train_sum_recall / folds))
    print("F-measure = %f" % (train_sum_fmeasure / folds))
    print("Sensitivity = %f" % (train_sum_sensitivity / folds))
    print("Specificity = %f" % (train_sum_specificity / folds))
    print('\nMax Accuracy = ', train_max_accuracy)

    print("\n\nAccuracy = %f" % (test_sum_accuracy / folds))
    print("Precision = %f" % (test_sum_precision / folds))
    print("Recall = %f" % (test_sum_recall / folds))
    print("F-measure = %f" % (test_sum_fmeasure / folds))
    print("Sensitivity = %f" % (test_sum_sensitivity / folds))
    print("Specificity = %f" % (test_sum_specificity / folds))
    print('\nMax Accuracy = ', test_max_accuracy)


def main():
    # Load our data and create our data set.
    print('Loading data...')
    DATA_COLUMNS = ['Date',
                    'HH',
                    'MM',
                    'SS',
                    'F2-F4[uV]',
                    'F4-C4[uV]',
                    # 'C4-P4[uV]',
                    # 'P4-O2[uV]',
                    # 'F1-F3[uV]',
                    # 'F3-C3[uV]',
                    # 'C3-P3[uV]',
                    # 'P3-O1[uV]',
                    'C4-A1[uV]',
                    'ECG1-ECG2[uV]'
                    ]
    data = signals.load('../data/n1.csv', DATA_COLUMNS)  # , 900000)

    print('Loading labels...')
    LABEL_COLUMNS = ['Time [hh:mm:ss]',
                     'Event'
                     ]
    labels = labels_script.load('../data/n1.txt', LABEL_COLUMNS)

    print('Creating data-set...')
    data_set = dataset.create(data, labels)
    describe(data_set)

    # Convert labels from strings to integers and define the number of classes (2 in our case)
    data_set, num_classes = convert_string_to_integer(data_set)

    cross_validation(10, data_set, num_classes)


if __name__ == '__main__':
    starttime = datetime.now()
    setup()
    main()
    print(datetime.now() - starttime)

