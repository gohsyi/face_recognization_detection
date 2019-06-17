import numpy as np

from common.argparser import args
from common.util import get_data_and_labels, get_logger, timed


with timed('loading data ...'):
    if args.model == 'cnn':
        X_train, y_train = get_data_and_labels('data/train.txt', False)
        X_test, y_test = get_data_and_labels('data/test.txt', False)
    else:
        X_train = np.loadtxt('data/hog_X_train.csv', delimiter=',')
        y_train = np.expand_dims(np.loadtxt('data/hog_y_train.csv', delimiter=','), -1)
        X_test = np.loadtxt('data/hog_X_test.csv', delimiter=',')
        y_test = np.expand_dims(np.loadtxt('data/hog_y_test.csv', delimiter=','), -1)


if args.model == 'lr':
    from baselines.logistic_regression import LogisticRegression
    logger = get_logger('lr_langevin' if args.langevin else 'lr')
    model = LogisticRegression(
        learning_rate=args.lr,
        total_epoches=args.total_epoches,
        langevin=args.langevin,
        seed=args.seed,
        logger=logger,
        X_test=X_test,
        y_test=y_test,
    )
    # from sklearn.linear_model.logistic import LogisticRegression
    # model = LogisticRegression()

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
        X_test=X_test,
        y_test=y_test,
    )

else:
    raise NotImplementedError

logger.info(args)

with timed('training ...'):
    model.fit(X_train, y_train)

with timed('predicting ...'):
    y_pred = model.predict(X_test)

assert y_pred.shape == y_test.shape
logger.info(f'acc:{np.mean(y_pred==y_test)}')
