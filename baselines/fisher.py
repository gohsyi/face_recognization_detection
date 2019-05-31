import heapq
import numpy as np

from tqdm import tqdm

from baselines.common import squared_difference
from baselines.common import Model


class Scatter(object):
    def __init__(self, X):
        self.X = X
        self.n = X.shape[0]
        self.mu = np.reshape(np.mean(X, axis=0), (-1, 1))  # mean vector
        self.sigma = np.reshape(np.std(X, axis=0), (-1, 1))  # std vector
        self.Sigma = np.matmul(self.sigma, self.sigma.T)  # std matrix


class LDA(Model):
    def __init__(self, N=2, K=5, logger=None, seed=0):
        """
        :param n: number of Eigenvectors we select
        """
        super(LDA, self).__init__(logger=logger, seed=seed)

        self.N = N
        self.knn = KNN(K)

    def fit(self, X, y):
        pos = Scatter(X[y==1, :])
        neg = Scatter(X[y==0, :])

        S_B = np.matmul((pos.mu-neg.mu), (pos.mu-neg.mu).T)
        S_W = pos.n * pos.Sigma + neg.n * neg.Sigma

        # Compute the Eigenvalues and Eigenvectors of SW^-1 x SB
        eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))

        # Select the two largest eigenvalues
        eigen_pairs = [[np.abs(eigval[i]), eigvec[:, i]] for i in range(len(eigval))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        self.beta = np.hstack([eigen_pairs[i][1][:, np.newaxis].real for i in range(self.N)])

        X = np.concatenate([np.matmul(X, self.beta), np.reshape(y, (-1, 1))], -1)
        self.knn.fit(X)

        return self

    def predict(self, X):
        return self.knn.predict(np.matmul(X, self.beta))


class KNN(object):
    def __init__(self, K=5):
        self.K = K

    def fit(self, X):
        self.X = X

    def predict(self, X):
        pred = []
        for x in tqdm(X):
            vote = [0, 0]

            neighbors = heapq.nsmallest(self.K, self.X, key=lambda xx: squared_difference(xx[:-1], x))
            for xx in neighbors:
                vote[int(xx[-1])] += 1 / squared_difference(xx[:-1], x)
            pred.append(np.argmax(vote))
        return pred


if __name__ == '__main__':
    from common.util import get_data_and_labels

    X_train, y_train = get_data_and_labels('data/train.txt')
    X_test, y_test = get_data_and_labels('data/test.txt')

    model = LDA()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('acc:%.4f' % np.mean(pred == y_test))
