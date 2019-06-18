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
parser.add_argument('-batch_size', type=int, default=256)

# detection setting
parser.add_argument('-winSize', type=int, default=200)
parser.add_argument('-stride', type=int, default=10)
parser.add_argument('-thres_score', type=float, default=0.95)
parser.add_argument('-thres_iou', type=float, default=0.1)
parser.add_argument('-scales', type=str, default='1.0,0.6,1.3')
parser.add_argument('-hw_ratios', type=str, default='1.0,1.2,1.4,1.3')

args = parser.parse_args()
args.scales = list(map(float, args.scales.split(',')))
args.hw_ratios = list(map(float, args.hw_ratios.split(',')))


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
