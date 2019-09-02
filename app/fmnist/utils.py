import io
import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from sklearn.metrics import confusion_matrix


class Metrics:

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = Metrics.precision_m(y_true, y_pred)
        recall = Metrics.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class TensorBoardConfusionMatrixCallback(keras.callbacks.Callback):
    __channel = 4

    def __init__(self, logdir, tag, X_val, y_val, labels_dict):
        super().__init__()
        self.__logdir = logdir
        self.__tag = tag
        self.__X_val = X_val
        self.__y_val = y_val
        self.__labels = list(labels_dict.values())


    def __plot_confusion_matrix(self, cm, class_names, epoch):

        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix-{}'.format(epoch))
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.close()

        return figure

    def __fig_to_image(self, fig):

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_string = buf.getvalue()
        im = Image.open(buf)
        # im.show()
        buf.close()
        height, width = im.size

        return image_string, height, width

    def on_epoch_end(self, epoch, logs={}):

        predictions = self.model.predict(self.__X_val)
        y_preds = np.argmax(predictions, axis=1).astype(int)
        y_true = np.argmax(self.__y_val, axis=1).astype(int)

        cm = confusion_matrix(y_true=y_true, y_pred=y_preds)

        figure = self.__plot_confusion_matrix(cm, self.__labels, epoch)

        image_string, height, width = self.__fig_to_image(figure)

        image = tf.Summary.Image(height=height, width=width, colorspace=self.__channel,
                                 encoded_image_string=image_string)

        summary = tf.Summary(value=[tf.Summary.Value(tag=self.__tag, image=image)])
        writer = tf.summary.FileWriter(self.__logdir)
        writer.add_summary(summary, epoch)
        writer.flush()
        return
