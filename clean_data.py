# Made by Jakub Wlodarczyk
# Clean up our data so it creates a better model.

import numpy as np
import time
from random import shuffle

forward       = [1,0,0,0,0,0,0]
back          = [0,1,0,0,0,0,0]
left          = [0,0,1,0,0,0,0]
right         = [0,0,0,1,0,0,0]
forward_left  = [0,0,0,0,1,0,0]
forward_right = [0,0,0,0,0,1,0]
noop          = [0,0,0,0,0,0,1]

forward_count = 0
other_count = 0
forward_list = []
other_list = []
result_list = []

load_file_name = 'D:\\Projects\\SelfDrivingFZero\\trainingData.npy'
save_file_name = 'D:\\Projects\\SelfDrivingFZero\\cleanTrainingData.npy'

data = np.load(load_file_name)
shuffle(data)

for row in data:
    if row[1] == forward:
        forward_count += 1
        forward_list.append(row)
    else:
        other_count += 1
        other_list.append(row)

if forward_count > other_count:
    result_list = forward_list[:(other_count*2)] + other_list
else:
    result_list = forward_list + other_list

print('Number of forward inputs: ' + str(forward_count))
print('Number of other inputs: ' + str(other_count))
print('Length of final data list : ' + str(len(result_list)))

if result_list:
    np.save(save_file_name, result_list)