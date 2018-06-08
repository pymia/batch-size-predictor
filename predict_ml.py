"""aaa"""
from numpy import genfromtxt
import numpy as np
from sklearn.naive_bayes import BernoulliNB


def read_csv(path):
    """aaa"""
    return genfromtxt(path, delimiter=',')


class Classifier(object):
    """BernoulliNB"""

    def __init__(self):
        clf = BernoulliNB()
        train_x = np.random.randint(2, size=(6, 10))
        train_y = np.array([1, 2, 3, 4, 4, 5])
        self.model = clf.fit(train_x, train_y)

    def predict(self, test_arr):
        """predict with input"""
        return self.model.predict(test_arr)


def main():
    """define model and predict"""
    model = Classifier()
    test_arr = [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]
    predict_arr = model.predict(test_arr)
    print(predict_arr)


if __name__ == "__main__":
    main()
