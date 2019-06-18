import numpy as np


class LDA:
    def __init__(self):
        self.beta = np.zeros(900)
        self.pos = 0
        self.neg = 0

    def fit(self, X, y):
        y = np.squeeze(y)
        x_pos = X[y==1]
        x_neg = X[y==0]
        avg_pos = x_pos.mean(0)
        avg_neg = x_neg.mean(0)

        SB = np.matmul(np.mat(avg_pos - avg_neg).T, np.mat(avg_pos - avg_neg))
        SW = np.matmul(np.mat(avg_pos - x_pos).T, np.mat(avg_pos - x_pos)) + \
             np.matmul(np.mat(avg_neg - x_neg).T, np.mat(avg_neg - x_neg))
        S = np.matmul(np.linalg.inv(SW), SB)
        eig_val, eig_vec = np.linalg.eig(S)
        self.beta = eig_vec.T[0]
        self.pos = np.matmul(np.mat(avg_pos), self.beta.T)
        self.neg = np.matmul(np.mat(avg_neg), self.beta.T)

    def predict(self, X):
        label = []
        XX = np.matmul(X, self.beta.T)
        for i in range(len(X)):
            if np.linalg.norm(XX[i] - self.pos) < np.linalg.norm(XX[i] - self.neg):
                label.append(1)
            else:
                label.append(0)

        label = np.array(label)
        x_pos = X[label==1]
        x_neg = X[label==0]

        avg_pos = x_pos.mean(0)
        avg_neg = x_neg.mean(0)

        self.inter = np.matmul(np.mat(avg_pos - avg_neg), self.beta.T)
        SW = np.matmul(np.mat(avg_pos - x_pos).T, np.mat(avg_pos - x_pos)) + \
             np.matmul(np.mat(avg_neg - x_neg).T, np.mat(avg_neg - x_neg))
        self.intra = np.square(np.matmul(np.mat(self.beta), np.matmul(SW, (np.mat(self.beta).T))))

        return label
