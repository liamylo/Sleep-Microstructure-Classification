import re
from datetime import datetime

import pandas as pd


def load(file_path, column_names):

    labels = _read(file_path, column_names)
    # Extract annotation start and end of recording dates.
    recording_dates = _get_recording_dates(file_path)
    labels = _change_datetime_structure(labels, recording_dates)
    labels['Event'] = convert_label_names(labels['Event'])

    return labels


def _read(file_path, column_names):
    labels = pd.read_csv(file_path,
                         skiprows=21,
                         sep='\t',
                         usecols=column_names)

    return labels


def _get_recording_dates(file_path):
    labels = pd.read_csv(file_path, sep='/t', nrows=20, engine='python')

    match_start = re.search(r'\d{2}/\d{2}/\d{4}', str(labels.iloc[2].values))
    match_end = re.search(r'\d{2}/\d{2}/\d{4}', str(labels.iloc[-1].values))

    start_date = datetime.strptime(match_start.group(), '%d/%m/%Y').strftime('%Y/%m/%d')
    end_date = datetime.strptime(match_end.group(), '%d/%m/%Y').strftime('%Y/%m/%d')

    return {'start_date': start_date, 'end_date': end_date}


def _change_datetime_structure(labels, recording_dates):

    # Combine the recording dates with the Time [hh:mm:ss] in one column named Datetime.
    datetime_ = _concatenate_datetime(labels['Time [hh:mm:ss]'], recording_dates)

    # Insert column Datetime in the first column of the DataFrame
    labels.insert(loc=0, column='Datetime', value=datetime_)

    # Delete column 'Time [hh:mm:ss]' because it is no longer needed,
    # and is replaced by the column: Datetime
    labels.drop(columns=['Time [hh:mm:ss]'], axis=1, inplace=True)

    return labels


def _concatenate_datetime(time_, recording_dates):
    datetime_ = []
    date = recording_dates['start_date']
    midnight = '00:00:00'

    for i in range(len(time_)):
        if time_[i].split(':')[0] == midnight.split(':')[0] and date != recording_dates['end_date']:
            date = recording_dates['end_date']

        datetime_.append("{} {}".format(date, str(time_[i])))

    datetime_ = pd.Series(pd.to_datetime(datetime_, format='%Y/%m/%d %H:%M:%S.%f'), index=None) \
        .dt.strftime('%Y%m%d%H%M%S')

    return datetime_


def convert_label_names(events):
    # Change the label names to have only CAP and NON-CAP phases.
    events = events.replace(to_replace=r'^M.*', value='CAP', regex=True)
    events = events.replace(to_replace=r'^S.*', value='NON-CAP', regex=True)

    return events
