import os, sys, time, logging
import cv2

from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

from skimage.feature import hog
from contextlib import contextmanager
from skimage import exposure


loggerDict = []

def get_logger(name, folder='logs'):
    """
    returns a logger with specific name and write logs to folder/name.log
    :param folder: the folder containing log file
    :param name: name of this logger
    :return: created logger
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

    if name in loggerDict:
        return logging.getLogger(name)

    loggerDict.append(name)

    # if os.path.exists(os.path.join('logs', '{}.log'.format(name))):
    #     os.remove(os.path.join('logs', '{}.log'.format(name)))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\tmodel:{}\t%(message)s'.format(name))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(os.path.join(folder, '{}.log'.format(name)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(msg, logger=None):
    """
    a context manager for timing
    :param msg: what msg you want to display
    :param logger: if not None, use this logger to output instead of print
    :return: None
    """

    output = logger.info if logger is not None else print
    tstart = time.time()

    output(msg)
    yield
    output('done in %.1f seconds' % (time.time() - tstart))


def get_data_and_labels(txt, do_hog=True):
    """
    get all data and labels from .txt file
    :param txt: txt file containing all the pathes of data and label, the format has to be:
                <path_to_the_image> <0/1>
    :returns data, label: array of shape [num_of_samples, num_of_features] and [num_of_samples, 1]
    """

    X, y = [], []
    data_and_labels = open(txt).readlines()

    with timed(f'getting all data and labels from {txt}'):
        time.sleep(0.5)
        for line in tqdm(data_and_labels):
            image_path, label = line.split()
            image = cv2.imread(image_path)
            image = cv2.resize(image, (96, 96))

            if do_hog:
                image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = hog(
                    image_grey,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                )

            X.append(image)
            y.append(int(label))

        time.sleep(0.5)

    ## visualization
    if do_hog:
        image_path, label = data_and_labels[0].split()
        image = cv2.imread(image_path)
        image = cv2.resize(image, (96, 96))
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex='all', sharey='all')
        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # original image is BGR
        ax1.set_title('Original image')

        fd, hog_image = hog(
            image_grey,
            visualise=True,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
        )

        ### Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap='gray')
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return shuffle(np.array(X), np.array(y))


def shuffle(X, y):
    """
    shuffle data
    :param X: feature vector
    :param y: label vector
    :return: shuffled feature vector, shuffled label vector
    """

    assert X.shape[0] == y.shape[0]

    if len(X.shape) > 2:
        return np.array(X), np.array(y)

    if len(y.shape) == 1:
        y = np.reshape(y, (-1, 1))

    data = np.concatenate([X, y], -1)
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]

    return X, y
