import sys, os, datetime
sys.path.insert(1, os.path.join(os.getcwd(), "src/models"))
from dynamic_model import DynamicModel
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

"""
Author: Andrey Bulezyuk @ German IT Academy (https://git-academy.com)
Date: 18.01.2020
"""

class Service():

    # model_name must be supplied. 
    # otherwise no configuration cad be loaded.
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.dynamic_model = DynamicModel(self.model_name)

    def _get_train_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # reshape to be [samples][width][height][channels]
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
        
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        # Load data
        self._get_train_data()

        # This return the compiled Keras Model from dynamic_model->model()
        print(self.y_train)
        model = self.dynamic_model.model()
        model.fit(self.x_train, self.y_train,
            batch_size=1000,
            epochs=4,
            verbose=1) 

        # Save trained model
        now = datetime.datetime.now()
        model.save(f"src/models/{self.model_name}_{now.year}{now.month}{now.day}_{now.hour}{now.minute}.h5")
        return True


    def predict(self, X):
        # Load model
        model = self._load_model()
        
        # Execute
        results = model.predict(X)
        if results is not None and results != False:
            return results
        return False