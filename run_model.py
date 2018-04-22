import numpy as np
import cv2
import time
import random
from pynput.keyboard import Key, Controller
from models import alexnet
from collections import deque, Counter
from PIL import ImageGrab, Image
from mss import mss
from statistics import mode,mean

bbox = (0, 175, 501, 350)
sct = mss()

keyboard = Controller()

WIDTH = 175
HEIGHT = 501
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)

forward       = [1,0,0,0,0,0,0]
back          = [0,1,0,0,0,0,0]
left          = [0,0,1,0,0,0,0]
right         = [0,0,0,1,0,0,0]
forward_left  = [0,0,0,0,1,0,0]
forward_right = [0,0,0,0,0,1,0]
noop          = [0,0,0,0,0,0,1]

def key_forward():
    keyboard.press(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.left)
    keyboard.release(Key.right)

def key_left():
    keyboard.press(Key.left)
    keyboard.release(Key.down)
    keyboard.release(Key.up)
    keyboard.release(Key.right)

def key_right():
    keyboard.press(Key.right)
    keyboard.release(Key.down)
    keyboard.release(Key.up)
    keyboard.release(Key.left)
    
def key_back():
    keyboard.press(Key.down)
    keyboard.release(Key.right)
    keyboard.release(Key.up)
    keyboard.release(Key.left)

def key_forward_left():
    keyboard.press(Key.up)
    keyboard.press(Key.left)
    keyboard.release(Key.down)
    keyboard.release(Key.right)

def key_forward_right():
    keyboard.press(Key.up)
    keyboard.press(Key.right)
    keyboard.release(Key.down)
    keyboard.release(Key.left)

def key_no_op():
    keyboard.release(Key.up)
    keyboard.release(Key.right)
    keyboard.release(Key.down)
    keyboard.release(Key.left)
    
model = alexnet(WIDTH, HEIGHT, LR, output=7)
MODEL_NAME = 'model_v2'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')

def main():
    last_time = time.time()

    paused = True
    mode_choice = 0

    while(True):
        img = sct.grab(bbox)
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 120)
        cv2.imshow('window', edges)
        
        if not paused:
            prediction = model.predict([edges.reshape(WIDTH,HEIGHT,1)])[0]
            prediction = np.array(prediction) * np.array([0.04, 5.2, 2.2, 7.2, 7.2, 16.3, 6.0])
            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                key_forward()
                choice_picked = 'forward'               
            elif mode_choice == 1:
                key_back()
                choice_picked = 'back'
            elif mode_choice == 2:
                key_left()
                choice_picked = 'left'
            elif mode_choice == 3:
                key_right()
                choice_picked = 'right'
            elif mode_choice == 4:
                key_forward_left()
                choice_picked = 'forwardLeft'
            elif mode_choice == 5:
                key_forward_right()
                choice_picked = 'forwardRight'
            elif mode_choice == 6:
                key_no_op()
                choice_picked = 'no-op'

            print('Choice: {}'.format(choice_picked))


        key = cv2.waitKey(25) & 0xFF

        if key == ord('w'):
            paused = not paused
            print("Paused" if paused else "Unpaused")
            if not paused:
                time.sleep(1)
                
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

main()       