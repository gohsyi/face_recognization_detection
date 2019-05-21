import os
import sys
import cv2
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from tqdm import tqdm
from skimage.feature import hog
from contextlib import contextmanager
from skimage import exposure


"""
returns an empty list with shape (d0, d1, d2)
"""
def empty_list(d0, d1=None, d2=None):
    if d1 is None:
        return [[] for _ in range(d0)]
    elif d2 is None:
        return [[[] for _ in range(d1)] for __ in range(d0)]
    else:
        return [[[[] for _ in range(d2)] for __ in range(d1)] for ___ in range(d0)]


"""
returns a logger with std output and file output
"""
def get_logger(folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)

    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    if os.path.exists(os.path.join('logs', '{}.log'.format(name))):
        os.remove(os.path.join('logs', '{}.log'.format(name)))

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


"""
timing function
"""
@contextmanager
def timed(msg, logger=None):
    tstart = time.time()
    yield
    if logger is not None:
        logger.info(f'{msg} done in %.1f seconds' % (time.time() - tstart))
    else:
        print(f'{msg} done in %.1f seconds' % (time.time() - tstart))


"""
get all data and labels from .txt file

format:
<path_to_the_image> <0/1>
"""
def get_data_and_labels(txt):
    X, y = [], []
    data_and_labels = open(txt).readlines()

    with timed(f'get all data and labels from {txt}'):
        for line in tqdm(data_and_labels):
            image_path, label = line.split()
            image = cv2.imread(image_path)
            image = cv2.resize(image, (96, 96))
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fd = hog(
                image_grey,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
            )
            X.append(fd)
            y.append(int(label))

        time.sleep(0.5)

    ## visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
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

    return np.array(X), np.array(y)
