from baselines.common.misc_util import set_global_seeds


class Model(object):
    def __init__(self, logger=None, seed=0):
        self.output = logger.info if logger else print
        set_global_seeds(seed)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
