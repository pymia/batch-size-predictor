"""
predict batch size with defined Classifier
version: neural network
"""
import os.path
import argparse
import numpy as np
from numpy import genfromtxt
from numpy import argmax
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot
MODEL_PATH = "my_model.h5"
TRAIN_PATH = "train_cases.csv"
RESULT_PATH = "result.csv"


def read_csv(path):
    """read csv file with genfromtxt to get numpy arr"""
    return genfromtxt(path, delimiter=',')


def save_as_file(predict_arr):
    """save_as_file"""
    np.savetxt(RESULT_PATH, predict_arr, fmt='%d', newline='\n')


def base_model():
    """model component"""
    model = Sequential()
    model.add(Dense(32, input_dim=6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


class Classifier(object):
    """Keras NN Classifier"""

    def __init__(self):
        if os.path.isfile(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
        else:
            train_dataset = read_csv(TRAIN_PATH)
            dataset_x = [i[:6] for i in train_dataset]
            dataset_y = [i[-1] for i in train_dataset]
            dataset_y = np_utils.to_categorical(dataset_y, 7)
            train_x, test_x, train_y, test_y = train_test_split(
                dataset_x, dataset_y, test_size=0.33, random_state=42)
            self.model = base_model()
            history = self.model.fit(np.asarray(train_x), np.asarray(train_y), batch_size=256,
                                     nb_epoch=3000, verbose=2)
            self.model.save(MODEL_PATH)

    def predict(self, path):
        """predict with input"""
        test_arr = read_csv(path)
        result = self.model.predict(test_arr)
        return [argmax(i) for i in result]


def main(path):
    """define model and predict"""
    model = Classifier()
    predict_arr = model.predict(path)
    save_as_file(predict_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for ml preditor')
    parser.add_argument('path', help='test csv path')
    args = parser.parse_args()
    main(args.path)
