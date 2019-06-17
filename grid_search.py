import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from scripts import models


# TODO Use Grid Search for hyper parameter optimization!
# Hyper parameters:
seed = 7
np.random.seed(seed)
BATCH_SIZE = [10, 20, 32, 40, 60, 80, 100]
EPOCHS = [10, 50, 100]
OPTIMIZER = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
INIT_MODE = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
ACTIVATION = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
WEIGHT_CONSTRAINT = [1, 2, 3, 4, 5]
DROPOUT_RATE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KERNEL_SIZE = [3, 5, 7]
FILTERS = [50, 100, 150, 200]
POOL_SIZE = []


def grid_search(TIME_PERIODS, input_shape, num_sensors, num_classes, x_train, y_train_hot):
    # TODO Define the model
    model = KerasClassifier(build_fn=models.cnn2,
                            time_periods=TIME_PERIODS,
                            input_shape=input_shape,
                            number_of_sensors=num_sensors,
                            number_of_classes=num_classes,
                            verbose=0,
                            # epochs=50,
                            # batch_size=80
                            )

    # TODO HERE CHANGE HYPER PARAMETERS:
    param_grid = dict(epochs=EPOCHS, batch_size=BATCH_SIZE)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(x_train, y_train_hot)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))