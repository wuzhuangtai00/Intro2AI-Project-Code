import re
import os
from utils import makedirs
import matplotlib.pyplot as plt

file_name = '../log/mnist_std/train_log.txt'
affix = 'std'
title = r'Standard Training'

img_folder = '../img'
makedirs(img_folder)

train_iter_list = []
train_acc_list = []
train_rob_list = []

test_iter_list = []
test_acc_list = []
test_rob_list = []

output_margin = []
all_layer_margin = []

epoch = 0


with open(file_name, 'r') as f:
    lines = f.readlines()

    for line in lines:
        splits = re.split('[, =%:\n]+', line)

        if splits[0] == 'epoch':
            _iter = int(splits[3])
            train_iter_list.append(_iter)

        if splits[0] == 'standard':
            train_acc_list.append(float(splits[2]))
            train_rob_list.append(float(splits[5]))

        if splits[0] == 'test':
            print(splits)
            epoch += 1
            test_iter_list.append(epoch)
            test_acc_list.append(float(splits[2]))
            train_acc_list.append(float(splits[9]))
            test_rob_list.append(float(splits[6]))
            output_margin.append(float(splits[19]))

print(output_margin)

a_1 = plt.plot(test_iter_list, train_acc_list , color='r', label='train standard accuary')[0]
a_2 = plt.plot(test_iter_list, test_acc_list , color='r', linestyle='--', label='test standard accuary')[0]

# b_1 = plt.plot(train_iter_list, train_rob_list , color='b', label='train robust accuary')[0]
b_2 = plt.plot(test_iter_list, test_rob_list , color='b', linestyle='--', label='test robust accuary')[0]

c_1 = plt.plot(test_iter_list, output_margin , color='g', linestyle='--', label='output margin')[0]
c_1 = plt.plot(test_iter_list, all_layer_margin , color='g', linestyle='--', label='all-layer margin')[0]

plt.title(title)

plt.legend(handles=[a_1, a_2, b_1, b_2])

plt.savefig(os.path.join(img_folder, 'mnist_learning_curve_%s.jpg' % affix))