from fmnist.fmnistexperiment import FMNistBuilderParameters, FMnistExperiment
import warnings

from fmnist.utils import Metrics

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Label of classes
    labels_dict = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    # Metrics: Accuracy, Precision, Recall, F1
    metrics = ['accuracy', Metrics.precision_m, Metrics.recall_m, Metrics.f1_m]

    # Creating an object instance of FMNistBuilderParameters to define module parameters.
    parameters = FMNistBuilderParameters() \
        .with_file_paths('../data/fashion-mnist_train.csv',
                         '../data/fashion-mnist_test.csv') \
        .with_image_size(28, 28) \
        .with_callback_params('val_loss', 'min', 10) \
        .with_train_params(batch_size=2560,
                           epochs=50,
                           labels_dict=labels_dict,
                           test_size=0.2,
                           random_state=13,
                           metrics=metrics) \
        .build()

    # Passing parameters to the module
    experiment = FMnistExperiment(parameters)

    # Traing the experiment.
    # Epoch and metrics(accuracy, f1, precision, recall) logs are dumping over TensorBoard and MLFlow servers.
    # For tensorboard, http://localhost:6006/
    # For MLFlow server, http://localhost:5000/
    experiment.train()

    # Testing dataset over the model
    test_scores = experiment.evaluate()
    print(test_scores)
