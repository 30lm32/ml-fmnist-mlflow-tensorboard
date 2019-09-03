import os
from datetime import datetime

import keras
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from fmnist.utils import TensorBoardConfusionMatrixCallback


# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class FMNistBuilderParameters(object):
    class Parameters(object):
        def __init__(self):
            self.file_path_train = None
            self.file_path_test = None
            self.img_height = None
            self.img_width = None
            self.batch_size = None
            self.labels_dict = None
            self.number_classes = None
            self.epochs = None
            self.test_size = None
            self.random_state = None
            self.metrics = None
            self.early_stopping_monitor = None
            self.early_stopping_patience = None
            self.early_stopping_mode = None

    def __init__(self):
        self.__parameters = FMNistBuilderParameters.Parameters()

    def with_file_paths(self, file_path_train, file_path_test):
        self.__parameters.file_path_train = file_path_train
        self.__parameters.file_path_test = file_path_test
        return self

    def with_image_size(self, img_height, img_width):
        self.__parameters.img_height = img_height
        self.__parameters.img_width = img_width
        return self

    def with_callback_params(self, early_stopping_monitor='val_loss',
                             early_stopping_mode='min',
                             early_stopping_patience=10):
        self.__parameters.early_stopping_monitor = early_stopping_monitor
        self.__parameters.early_stopping_mode = early_stopping_mode
        self.__parameters.early_stopping_patience = early_stopping_patience

        return self

    def with_train_params(self,
                          batch_size,
                          epochs,
                          labels_dict,
                          test_size,
                          random_state,
                          metrics):
        self.__parameters.batch_size = batch_size
        self.__parameters.epochs = epochs
        self.__parameters.labels_dict = labels_dict
        self.__parameters.number_classes = len(labels_dict)
        self.__parameters.test_size = test_size
        self.__parameters.random_state = random_state
        self.__parameters.metrics = metrics
        return self

    def build(self) -> Parameters:
        return self.__parameters


class FMnistExperiment:
    __model = None
    __callbacks = []
    __mlflow_keras_model_log = False
    __tensorboard_callback = None

    __format_logdir_images_cm = '../{}/images/cm/{}/'
    __format_logdir_scalars = '../{}/scalars/{}/'
    __tensorboard_log_dir = os.getenv('TENSORBOARD_LOGS_DIR')

    def __init__(self, params: FMNistBuilderParameters.Parameters):

        self.__params = params

        self.__init_mlflow_log()

        self.data_train = pd.read_csv(self.__params.file_path_train)
        self.data_test = pd.read_csv(self.__params.file_path_test)
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = self.__split_data()

        self.__init_callbacks()
        self.__create_model()

    def __init_tensorboard_callbacks(self):

        tensorboard_scalar_logs = os.getenv('TENSORBOARD_SCALAR_LOGS')
        if tensorboard_scalar_logs is not None and tensorboard_scalar_logs:
            now = datetime.now()
            logdir = self.__format_logdir_scalars.format(self.__tensorboard_log_dir, now.strftime("%Y%m%d-%H%M%S"))
            tensorboard_scalar_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            self.__callbacks.append(tensorboard_scalar_callback)

        tensorboard_confusion_matrix = os.getenv('TENSORBOARD_CONFUSION_MATRIX')
        if tensorboard_confusion_matrix is not None and tensorboard_confusion_matrix:
            now = datetime.now()
            logdir = self.__format_logdir_images_cm.format(self.__tensorboard_log_dir, now.strftime("%Y%m%d-%H%M%S"))

            tensorboard_cm_callback = TensorBoardConfusionMatrixCallback(logdir,
                                                                         "ConfusionMatrix",
                                                                         self.X_val,
                                                                         self.y_val,
                                                                         self.__params.labels_dict)
            self.__callbacks.append(tensorboard_cm_callback)

    def __init_early_stopping_callback(self):

        if self.__params.early_stopping_monitor is not None and \
                self.__params.early_stopping_mode is not None and \
                self.__params.early_stopping_patience:
            early_stopping_callback = EarlyStopping(monitor=self.__params.early_stopping_monitor,
                                                    mode=self.__params.early_stopping_mode,
                                                    verbose=1,
                                                    patience=self.__params.early_stopping_patience)
            self.__callbacks.append(early_stopping_callback)

    def __init_callbacks(self):
        self.__init_tensorboard_callbacks()
        self.__init_early_stopping_callback()

    def __init_mlflow_log(self):

        self.__mlflow_keras_model_log = os.getenv('MLFLOW_KERAS_MODEL_LOG')
        if self.__mlflow_keras_model_log is None:
            self.__mlflow_keras_model_log = False

        mlflow_experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
        mlflow_keras_auto_log = os.getenv('MLFLOW_KERAS_AUTO_LOG')
        mlflow_log_params = os.getenv('MLFLOW_LOG_PARAMS')

        if mlflow_experiment_name is not None:
            mlflow.set_experiment(mlflow_experiment_name)

        if mlflow_keras_auto_log is not None and mlflow_keras_auto_log:
            mlflow.keras.autolog()

        if mlflow_log_params is not None and mlflow_log_params:
            mlflow.log_params(self.__params.__dict__)

    def __split_data(self):
        X = np.array(self.data_train.iloc[:, 1:])
        y = to_categorical(np.array(self.data_train.iloc[:, 0]))

        X_train, X_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          test_size=self.__params.test_size,
                                                          random_state=self.__params.random_state)

        # Test data
        X_test = np.array(self.data_test.iloc[:, 1:])
        y_test = to_categorical(np.array(self.data_test.iloc[:, 0]))

        X_train = X_train.reshape(X_train.shape[0],
                                  self.__params.img_height,
                                  self.__params.img_width,
                                  1)

        X_test = X_test.reshape(X_test.shape[0],
                                self.__params.img_height,
                                self.__params.img_width,
                                1)

        X_val = X_val.reshape(X_val.shape[0],
                              self.__params.img_height,
                              self.__params.img_width,
                              1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')
        X_train /= 255
        X_test /= 255
        X_val /= 255

        return X_train, y_train, X_test, y_test, X_val, y_val

    def __create_model(self):
        input_shape = (self.__params.img_height, self.__params.img_width, 1)

        self.__model = Sequential()

        self.__model.add(Conv2D(32, kernel_size=(3, 3),
                                activation='relu',
                                kernel_initializer='he_normal',
                                input_shape=input_shape))
        self.__model.add(BatchNormalization())
        self.__model.add(MaxPooling2D((2, 2)))
        self.__model.add(Dropout(0.25))

        self.__model.add(BatchNormalization())
        self.__model.add(Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Dropout(0.25))

        self.__model.add(BatchNormalization())
        self.__model.add(Conv2D(128, (3, 3), activation='relu'))
        self.__model.add(Dropout(0.4))

        self.__model.add(Flatten())
        self.__model.add(BatchNormalization())
        self.__model.add(Dense(128, activation='relu'))
        self.__model.add(Dropout(0.3))

        self.__model.add(Dense(self.__params.number_classes,
                               activation='softmax'))

        self.__model.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=keras.optimizers.Adam(),
                             metrics=self.__params.metrics)

        self.__model.summary()

    def train(self):

        if self.__model is not None:
            fit_params = {
                'x': self.X_train,
                'y': self.y_train,
                'batch_size': self.__params.batch_size,
                'verbose': 1,
                'epochs': self.__params.epochs,
                'validation_data': (self.X_val, self.y_val),
                'callbacks': self.__callbacks
            }

            history = self.__model.fit(**fit_params)
            if self.__mlflow_keras_model_log:
                mlflow.keras.log_model(self.__model, "models")

    def evaluate(self):

        if self.__model is not None:
            ev_params = {
                'x': self.X_test,
                'y': self.y_test,
                'batch_size': self.__params.batch_size,
                'verbose': 1
            }

            test_score_format = 'test_{}'
            test_metric_names = [test_score_format.format(m.__name__) if type(m).__name__ == 'function'
                                 else test_score_format.format(m)
                                 for m in self.__params.metrics]

            scores = self.__model.evaluate(**ev_params)
            d = dict(zip(test_metric_names, scores))
            for k, v in d.items():
                mlflow.log_metric(k, v)

            return d

        return None

    def predict(self, kwargs: dict):
        return self.__model.predict(**kwargs)
