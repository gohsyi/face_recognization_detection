import cv2
import numpy as np

from sklearn.svm import SVC
from skimage.feature import hog


class SVM():
    def __init__(self, kernel):
        self.backend = SVC(kernel=kernel)

    def fit(self, X, y):
        self.backend.fit(X, y)
        return self

    def predict(self, X):
        return self.backend.predict(X)

    def score(self, img):
        img = cv2.resize(img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = hog(
            img,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
        )

        return self.backend.decision_function(np.expand_dims(img, 0))
