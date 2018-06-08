"""
predict batch size with defined Classifier
version: machine learning
"""
import argparse
import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


TRAIN_PATH = "train_cases.csv"
RESULT_PATH = "result.csv"


def read_csv(path):
    """read csv file with genfromtxt to get numpy arr"""
    return genfromtxt(path, delimiter=',')


class Classifier(object):
    """DecisionTreeClassifier"""

    def __init__(self):
        clf = DecisionTreeClassifier()
        train_dataset = read_csv(TRAIN_PATH)
        dataset_x = [i[:6] for i in train_dataset]
        dataset_y = [i[-1] for i in train_dataset]
        train_x, test_x, train_y, test_y = train_test_split(
            dataset_x, dataset_y, test_size=0.33, random_state=42)
        self.model = clf.fit(train_x, train_y)

    def predict(self, path):
        """predict with input"""
        test_arr = read_csv(path)
        return self.model.predict(test_arr)


def save_as_file(predict_arr):
    np.savetxt(RESULT_PATH, predict_arr, fmt='%d', newline='\n')


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
