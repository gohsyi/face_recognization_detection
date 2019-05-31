import numpy as np

from common.argparser import args
from common.util import get_data_and_labels, get_logger, timed


do_hog = (args.model != 'cnn')

X_train, y_train = get_data_and_labels('data/train.txt', do_hog)
X_test, y_test = get_data_and_labels('data/test.txt', do_hog)


if args.model == 'lr':
    from baselines.logistic_regression import LogisticRegression
    logger = get_logger('lr_langevin' if args.langevin else 'lr')
    model = LogisticRegression(
        learning_rate=args.lr,
        total_epoches=args.total_epoches,
        langevin=args.langevin,
        seed=args.seed,
        logger=logger,
    )

elif args.model == 'svm':
    from sklearn.svm import SVC
    logger = get_logger(f'svm_{args.kernel}')
    model = SVC(kernel=args.kernel)

elif args.model == 'fisher':
    from baselines.fisher import LDA
    logger = get_logger('fisher')
    model = LDA(N=args.N, K=args.K, logger=logger, seed=args.seed)

elif args.model == 'cnn':
    from baselines.cnn import CNN
    logger = get_logger('cnn')
    model = CNN(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        total_epoches=args.total_epoches,
        seed=args.seed,
        logger=logger,
    )

else:
    raise NotImplementedError

with timed('training', logger=logger):
    model.fit(X_train, y_train)

with timed('predicting', logger=logger):
    y_pred = model.predict(X_test)


logger.info(f'acc:{np.mean(y_pred==y_test)}')
