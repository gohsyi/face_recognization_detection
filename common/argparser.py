import argparse


parser = argparse.ArgumentParser()

# gpu device
parser.add_argument('-gpu', type=str, default='-1')

# algorithm setting
parser.add_argument('-model', type=str, default='lr', help='lr/svm/fisher/cnn')

# common model setting
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-seed', type=int, default=0, help='global random seed')
parser.add_argument('-total_epoches', type=int, default=int(1e2))

# logistic regression setting
parser.add_argument('-langevin', action='store_true', default=False)

# svm setting
parser.add_argument('-kernel', type=str, default='linear', help='linear/rbf/poly')

# cnn setting
parser.add_argument('-batch_size', type=int, default=265)

# fisher setting
parser.add_argument('-N', type=int, default=20, help='choose N eigenvectors')
parser.add_argument('-K', type=int, default=5, help='K nearest neighbours')

args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
