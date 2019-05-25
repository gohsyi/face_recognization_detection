import numpy as np
import tensorflow as tf

from util import get_logger, get_data_and_labels

from baselines.common import layers
from baselines.common import tf_util


class CNN(object):
    """
    CNN model
    __init__:
    - initialize the CNN model

    fit():
    - fit the model with X_train and y_train

    predict():
    - predict y_test with X_test
    """

    def __init__(self,
                 learning_rate=1e-4,  # learning rate
                 batch_size=256,  # batch size
                 total_epoches=int(1e3),  # total number of epoches
                 image_height=96,  # height of image
                 image_width=96,  # width of image
                 image_channels=3,  # channels of image
                 log_interval=10):  # specifies how frequently the logs are printed out

        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels], 'observation')
        self.Y = tf.placeholder(tf.int32, [None], 'ground_truth')
        self.LR = tf.placeholder(tf.float32, [], 'learning_rate')

        conv1 = layers.conv2d(self.X, filters=16, ksize=(5, 5))
        pool1 = layers.max_pooling2d(conv1, psize=2, strides=2)

        conv2 = layers.conv2d(pool1, filters=32, ksize=(3, 3))
        pool2 = layers.max_pooling2d(conv2, psize=2, strides=2)

        probs = layers.dense(tf.layers.flatten(pool2), 2, activation=tf.nn.softmax)

        self.y_pred = tf.argmax(probs, -1)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=probs,
            labels=self.Y,
        ))

        lr = tf.train.polynomial_decay(
            learning_rate=learning_rate,
            global_step=tf.train.get_or_create_global_step(),
            decay_steps=total_epoches,
            end_learning_rate=learning_rate / 10,
        )

        self.trainer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.output = get_logger('cnn').info
        self.sess = tf_util.get_session()
        self.batch_size = batch_size
        self.total_epoches = total_epoches
        self.log_interval = log_interval

    def fit(self, X, y):
        """
        fit the model
        :param X: training data
        :param y: trainint label
        :return: self
        """

        self.sess.run(tf.global_variables_initializer())

        losses = []
        for ep in range(self.total_epoches):
            mb_x, mb_y = self.batch(X, y)
            loss, _ = self.sess.run([self.loss, self.trainer], feed_dict={
                self.X: mb_x,
                self.Y: mb_y,
            })
            losses.append(loss)

            if ep % self.log_interval == 0:
                self.output(f'ep:{ep}\tloss:%.4f' % np.mean(losses))
                losses = []

        return self

    def predict(self, X):
        """
        predict the label of X
        :param X: input data
        :return: predicted labels
        """

        return self.sess.run([self.y_pred], feed_dict={
            self.X: X
        })

    def batch(self, X, y):
        """
        generate a mini-batch
        :param X: training data
        :param y: training label
        :return: a mini-batch for training
        """

        batch_indices = np.random.choice(X.shape[0], self.batch_size)
        return X[batch_indices, :, :, :], y[batch_indices]


# test
if __name__ == '__main__':
    X, y = get_data_and_labels('data/train.txt', do_hog=False)
    model = CNN()
    model.fit(X, y)

    X, y = get_data_and_labels('data/test.txt', do_hog=False)
    pred = model.predict(X)
    print('acc:%.4f' % np.mean(pred == y))
