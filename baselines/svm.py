import numpy as np

from sklearn.svm import SVC


if __name__ == '__main__':
    kernel = 'poly'

    from common.util import get_data_and_labels
    X_train, y_train = get_data_and_labels('data/train.txt')

    model = SVC(kernel=kernel, verbose=True)
    model.fit(X_train, y_train)

    X_test, y_test = get_data_and_labels('data/test.txt')
    y_pred = model.predict(X_test)

    print('acc:%.4f' % np.mean(y_pred == y_test))
