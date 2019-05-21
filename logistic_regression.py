import numpy as np
from util import timed, get_data_and_labels
from sklearn.linear_model import LogisticRegression


class LR(object):
    def __init__(self):
        self.X, self.y = get_data_and_labels('train.txt')
        self.model = LogisticRegression(verbose=1)

        with timed('fit logistic regression'):
            self.model.fit(self.X, self.y)

    def test(self):
        test_X, test_y = get_data_and_labels('test.txt')

        with timed('predict'):
            prediction = self.model.predict(test_X)

        acc = np.mean(prediction == test_y)
        print('acc:%.3f' % acc)


if __name__ == '__main__':
    model = LR()
    model.test()
