import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")


parser = argparse.ArgumentParser()
parser.add_argument('-smooth', type=float, default=0)
args = parser.parse_args()


for root, dirs, files in os.walk('logs'):
    for f in files:
        if f[0] != '.' and f.split('.')[-1] == 'log':  # process .log
            p = os.path.join(root, f)
            print('processing %s' % p)
            loss = []
            acc = 0
            for line in open(p):
                line = line.split()
                for x in line:
                    x = x.split(':')
                    if x[0] == 'ep' and x[1] == '0':
                        loss = []
                        acc = 0
                    if x[0] == 'loss':
                        loss.append(float(x[1]))
                    if x[0] == 'acc':
                        acc = float(x[1])

            if len(loss) > 0:
                plt.plot(loss)
                plt.title('acc:%.2f' % acc)
                plt.savefig('.'.join(p.split('.')[:-1]) + '_loss.jpg')
                plt.cla()

