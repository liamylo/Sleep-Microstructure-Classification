import pandas as pd
import numpy as np
from scipy import stats


def create(data, labels):

    dataset = _labeling(data, labels)
    dataset = _fill_dataset(dataset)

    return dataset


def _labeling(data, labels):
    # Merge the signals and annotations together in order to give each signal the respective label.
    dataset = pd.merge_ordered(data, labels, left_by='Datetime')

    return dataset


def _fill_dataset(dataset):
    # Replace the empty cells of the 'Event' column with NaNs
    dataset[dataset['Event'] == ''] = np.NaN

    # Fill the empty cells of the 'Event' column with the previous value
    dataset = dataset.fillna(method='ffill')

    # Drop data that have no annotations
    dataset = dataset.dropna()

    return dataset


def create_segments_and_labels(dataset, time_steps, step, label_name):
    # signals as features
    N_FEATURES = dataset.shape[1] - 2
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # time_steps = 200, step = 100
    segments = []
    labels = []

    for i in range(0, len(dataset) - time_steps, step):
        f2_f4 = dataset['F2-F4[uV]'].values[i: i + time_steps]  # TODO Check what shape these return
        f4_c4 = dataset['F4-C4[uV]'].values[i: i + time_steps]
        c4_a1 = dataset['C4-A1[uV]'].values[i: i + time_steps]
        ecg1_ecg2 = dataset['ECG1-ECG2[uV]'].values[i: i + time_steps]

        # Retrieve the most often used label in this segment
        label = stats.mode(dataset[label_name][i: i + time_steps])[0][0]  # TODO Check which is the most used label
        segments.append([f2_f4, f4_c4, c4_a1, ecg1_ecg2])
        labels.append(label)

    # TODO Check what shape the segments have
    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    # TODO Check what the reshape returns
    labels = np.asarray(labels)

    return reshaped_segments, labels
