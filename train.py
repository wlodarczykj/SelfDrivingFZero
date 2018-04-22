import numpy as np
import cv2
import time
import os
from collections import deque
from models import alexnet
from random import shuffle

FILE_I_END = 1860

WIDTH = 175
HEIGHT = 501
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'model_v2'
PREV_MODEL = ''

forward       = [1,0,0,0,0,0,0]
back          = [0,1,0,0,0,0,0]
left          = [0,0,1,0,0,0,0]
right         = [0,0,0,1,0,0,0]
forward_left  = [0,0,0,0,1,0,0]
forward_right = [0,0,0,0,0,1,0]
noop          = [0,0,0,0,0,0,1]

model = alexnet(WIDTH, HEIGHT, LR, output=7)

for e in range(EPOCHS):
    #data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    try:
        file_name = "D:\Projects\SelfDrivingFZero\cleanTrainingData.npy"
        # full file info
        train_data = np.load(file_name)
        shuffle(train_data)
        train = train_data[:-50]
        test = train_data[-50:]

        print(str(train[0]))
        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)


        print('SAVING MODEL!')
        model.save(MODEL_NAME)
            
    except Exception as e:
        print(str(e))