import tensorflow as tf
from tensorflow.python.keras.constraints import maxnorm


def fully_connected(time_periods, input_shape, number_of_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((time_periods, 4), input_shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))
    print(model.summary())

    return model


def cnn1(time_periods, input_shape, number_of_sensors, number_of_classes,
         dropout_rate=0.5, optimizer='adam', activation='relu', weight_constraint=None,
         init_mode='glorot_uniform'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((time_periods, number_of_sensors), input_shape=(input_shape,)))
    model.add(tf.keras.layers.Conv1D(10, 10, input_shape=(time_periods, number_of_sensors),
                                     activation=activation, kernel_constraint=maxnorm(weight_constraint),
                                     kernel_initializer=init_mode))
    model.add(tf.keras.layers.Conv1D(10, 10, activation=activation, kernel_constraint=maxnorm(weight_constraint),
                                     kernel_initializer=init_mode))
    model.add(tf.keras.layers.MaxPooling1D(4))
    model.add(tf.keras.layers.Conv1D(16, 10, activation=activation, kernel_constraint=maxnorm(weight_constraint),
                                     kernel_initializer=init_mode))
    model.add(tf.keras.layers.Conv1D(16, 10, activation=activation, kernel_constraint=maxnorm(weight_constraint),
                                     kernel_initializer=init_mode))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc', 'mse'])

    return model


def cnn(time_periods, input_shape, number_of_sensors, number_of_classes, optimizer='adam', activation='relu', filters=100, kernel_size=10):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((time_periods, number_of_sensors), input_shape=(input_shape,)))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, input_shape=(time_periods, number_of_sensors), activation=activation))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation=activation))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation=activation))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation=activation))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc', 'mse'])

    return model


def cnn2(time_periods, input_shape, number_of_sensors, number_of_classes,
         optimizer='adam', activation='relu', init_mode='glorot_uniform',
         filters=100, kernel_size=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((time_periods, number_of_sensors), input_shape=(input_shape,)))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, input_shape=(time_periods, number_of_sensors),
                                     activation=activation, kernel_initializer=init_mode))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, input_shape=(time_periods, number_of_sensors),
                                     activation=activation, kernel_initializer=init_mode))
    model.add(tf.keras.layers.MaxPooling1D(4))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation=activation, kernel_initializer=init_mode))
    model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation=activation, kernel_initializer=init_mode))
    model.add(tf.keras.layers.MaxPooling1D(4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax', kernel_initializer=init_mode))
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
