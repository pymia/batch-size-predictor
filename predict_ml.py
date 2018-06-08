"""predict batch size with defined Classifier"""
"""version: machine learning"""

from numpy import genfromtxt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

TRAIN_PATH = "train_cases.csv"


def read_csv(path):
    """read csv file with genfromtxt to get numpy arr"""
    return genfromtxt(path, delimiter=',')


class Classifier(object):
    """BernoulliNB"""

    def __init__(self):
        clf = BernoulliNB()
        train_dataset = read_csv(TRAIN_PATH)
        dataset_x = [i[:6] for i in train_dataset]
        dataset_y = [i[-1] for i in train_dataset]
        train_x, test_x, train_y, test_y = train_test_split(
            dataset_x, dataset_y, test_size=0.33, random_state=42)
        self.model = clf.fit(train_x, train_y)
        self.test = test_x

    def predict(self):
        """predict with input"""
        test_arr = self.test
        return self.model.predict(test_arr)


def main():
    """define model and predict"""
    model = Classifier()
    predict_arr = model.predict()
    print(predict_arr)


if __name__ == "__main__":
    main()
