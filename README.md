# Fashion MNIST as Cloud Service

In this case, we used docker container technologies to create ML platform from scratch.
It is consists of four different docker containers (mlflow, notebook, postgres, tensorboard) that are already built in `docker-compose.yml`

The details of containers could be found under `./platform` directory.
Each container service has a specific dockerfile corresponding to the directories (mlflow, notebook, postgres, tensorboard) 
under platform directory

## Usage

1. Please, note that download the dataset (fashion-mnist_train.csv, fashion-mnist_test.csv) via the link below before staring use this project.

    https://www.kaggle.com/zalando-research/fashionmnist

2. Add the file under `data` directory.
3. You will find `makefile` to kick off the platform. It has three different commands to build, start, and stop platform.

* To build platform

        make build
    
* To start platform

        make start
    
* To stop platform

        make stop

# ML Model
We built CNN network to train dataset with four different metrics (accuracy, precision, recall, f1)
We are logging those metrics mlflow and tensorboard sides. In addition, we plotted confusion matrix on tensorboard side rather than notebook.

You may find the model we built in method `def __create_model` in `fmnist/fmnistexperiment.py`.
```python

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


```
# Docker Services

### Notebook
This service handles client side request via internet browser. We created a module (`fmnist`) for training and testing. 
Whenever we changed any codes in the module, Notebook are loading that module automatically macros in the first cell on the notebook.


![Image](./docs/image_nb_00.png)

### Tensorflow for TensorBoard
This service is being used to keep track confusion matrix and metrics' history of mode with respect to epoch. 
You may see the screenshots for TensorBoard, below.

![Image](./docs/image_tb_01.png)

![Image](./docs/image_tb_02.png)

![Image](./docs/image_tb_03.png)

![Image](./docs/image_tb_04.png)

![Image](./docs/image_tb_05.png)


### MLflow and Postgres
This service is being used to keep track and manage model parameters, metrics, model artifacts.
Postgres database is a database that is storing MLflow logs related to parameters, metrics.
Model artifacts is stored local disk. To manage flexibly storage, another options (S3, NFS etc.) could be used rather than local disk.
However, we preferred local disk in this implementation.

![Image](./docs/image_mf_01.png)

![Image](./docs/image_mf_02.png)

![Image](./docs/image_mf_03.png)

![Image](./docs/image_mf_04.png)

![Image](./docs/image_mf_05.png)

