import numpy as np
import numpy.linalg as lg


class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        y = np.squeeze(y)
        pos = X[y == 1, :]
        neg = X[y == 0, :]
        avg_pos = pos.mean(0)
        avg_neg = neg.mean(0)

        S_B = np.matmul(np.mat(avg_pos - avg_neg).T, np.mat(avg_pos - avg_neg))
        S_W = np.zeros((900, 900))

        for i in range(len(pos)):
            S_W = S_W + np.matmul(np.mat(avg_pos - pos[i]).T, np.mat(avg_pos - pos[i]))
        for i in range(len(neg)):
            S_W = S_W + np.matmul(np.mat(avg_neg - neg[i]).T, np.mat(avg_neg - neg[i]))

        S = np.matmul(lg.inv(S_W), S_B)
        eigenvalue, eigenvector = np.linalg.eig(S)
        self.beta = eigenvector.T[0]
        self.pos = np.matmul(avg_pos, self.beta)
        self.neg = np.matmul(avg_neg, self.beta)

    def predict(self, X):
        label = []
        X = np.matmul(X, self.beta)

        for i in range(len(X)):
            if np.linalg.norm(X[i] - self.pos) < np.linalg.norm(X[i] - self.neg):
                label.append(1)
            else:
                label.append(0)

        pos, neg = X[label==1], X[label==0]
        avg_pos = np.mean(pos, 0)
        avg_neg = np.mean(neg, 0)

        self.intra = np.matmul(np.mat(avg_pos - avg_neg), (self.beta).T)
        SW = np.zeros((900, 900))
        for i in range(len(pos)):
            SW = SW + np.matmul(np.mat(avg_pos - pos[i]).T, np.mat(avg_pos - pos[i]))
        for i in range(len(neg)):
            SW = SW + np.matmul(np.mat(avg_neg - neg[i]).T, np.mat(avg_neg - neg[i]))
        self.inter = np.matmul(np.mat(self.beta), np.matmul(SW, (np.mat(self.beta).T)))

        return label
