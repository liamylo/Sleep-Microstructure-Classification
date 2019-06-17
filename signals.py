import pandas as pd
import numpy as np


def load(file_path, column_names, nrows=None):

    data = _read(file_path, column_names, nrows)
    data = _resample(data)
    data = _change_datetime_structure(data)

    return data


def _read(file_path, column_names, nrows):
    if nrows is None:
        data = pd.read_csv(file_path,
                           sep='\t',
                           usecols=column_names
                           )
    else:
        data = pd.read_csv(file_path,
                           sep='\t',
                           usecols=column_names,
                           nrows=nrows
                           )

    return data


def _resample(data):

    frequency = 512 / 100

    resample = np.asarray(data)[0::int(frequency), :]

    resampled_data = pd.DataFrame(resample, columns=list(data))

    return resampled_data


def _change_datetime_structure(data):
    # Combine date and time, into a YYYY/MM/DD HH:MM:SS format and place them in one column.
    date_time = _create_datetime(data[['Date', 'HH', 'MM', 'SS']])

    # Insert column Datetime in the first column of the DataFrame
    data.insert(loc=0, column='Datetime', value=date_time)

    # Delete columns [Date, 'HH', 'MM', 'SS']
    data.drop(columns=['Date', 'HH', 'MM', 'SS'], axis=1, inplace=True)

    return data


def _create_datetime(datetime_):
    dates = datetime_.iloc[:, 0].values
    hours = datetime_.iloc[:, 1].values
    minutes = datetime_.iloc[:, 2].values
    seconds = datetime_.iloc[:, 3].values

    datetime_ = ["{} {}:{}:{}".format(date_, hour_, minute_, second_)
                 for date_, hour_, minute_, second_ in zip(dates, hours, minutes, seconds)]

    datetime_ = pd.Series(pd.to_datetime(datetime_, format='%Y/%m/%d %H:%M:%S.%f'),
                          index=None).dt.strftime('%Y%m%d%H%M%S')

    return datetime_
