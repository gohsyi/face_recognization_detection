import tensorflow as tf
import numpy as np
import random


def set_global_seeds(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
