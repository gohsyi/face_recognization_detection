import numpy as np

from baselines.common.model import Model

from common.util import get_data_and_labels, shuffle


class LogisticRegression(Model):
    """
    Logistic Regression Model
    * use __init__() to set learning rate and iterations of the model
    * use fit() to fit the model
    * use test(X, y) to test the accuracy
    """

    def __init__(self, learning_rate=0.0001, total_epoches=1000, langevin=0., logger=None, seed=0):
        super(LogisticRegression, self).__init__(logger=logger, seed=seed)

        self.learning_rate = learning_rate
        self.total_epoches = total_epoches
        self.langevin = langevin

        # set random seed
        np.random.seed(seed)

        # set output method
        self.output = logger.info if logger else print

    def fit(self, X, y):
        """
        fit the model
        :param X: training data
        :param y: trainint label
        :param opt: optimization method, 'sgd' or 'langevin'
        :return: self
        """

        # number of samples, number of features
        m, n = X.shape
        X, y = shuffle(X, y)
        y = np.reshape(y, (m, 1))

        # initialize parameter theta
        self.theta = np.random.normal(size=[n, 1])

        # use gradient ascent to fit this model
        costs = [[]]

        for ep in range(self.total_epoches):
            for xx, yy in zip(X, y):
                xx = np.reshape(xx, (1, n))
                yy = np.reshape(yy, (1, 1))
                self.gradient_ascent(xx, yy)

                costs[-1].append(self.cost(xx, yy))

            costs[-1] = float(np.mean(costs[-1]))
            self.output(f'ep:{ep}\tcost:%.3f' % costs[-1])

            costs.append([])

        return self

    def predict(self, X):
        """
        test our model, returns the accuracy
        :param X: test data
        :return: predicted labels
        """

        assert X.shape[1] == self.theta.shape[0]

        h = self.hypothesis(X)
        # h[h < 0.5] = -1
        h[h < 0.5] = 0
        h[h >= 0.5] = 1

        return h

    def cost(self, x, y):
        """
        compute the cost for visualization
        the cost is defined as `-y^T*log(h) - (1-y)^T*log(1-h)`
        :param x: x
        :param y: y
        :return: the cost
        """

        m = x.shape[0]
        h = self.hypothesis(x)
        return (np.matmul(-y.T, np.log(h)) + np.matmul(-(1-y).T, np.log(1-h))) / m

    def gradient_ascent(self, x, y):
        """
        use gradient descent to update theta
        partial neg-log-likelihood / partial theta_j = (h(x) - y) * x_j
        :param x: x
        :param y: y
        :param sigma: weight of langevin dynamics
        :return: None
        """

        m = x.shape[0]
        error = y - self.hypothesis(x)
        self.theta = self.theta + \
                     self.learning_rate * np.matmul(x.T, error) / m + \
                     self.langevin * np.sqrt(self.learning_rate) * np.random.normal(size=self.theta.shape)

    def hypothesis(self, x):
        return self.sigmoid(self.z(x))

    def z(self, x):
        return np.matmul(x, self.theta)

    @staticmethod
    def sigmoid(z):
        return 1. / (1 + np.exp(-z))


# test
if __name__ == '__main__':
    X, y = get_data_and_labels('data/train.txt')
    model = LogisticRegression(learning_rate=0.0005, total_epoches=100)
    model.fit(X, y)

    X, y = get_data_and_labels('data/test.txt')
    pred = model.predict(X)
    print('acc:%.4f' % np.mean(pred == y))
