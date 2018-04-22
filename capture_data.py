# Made by Jakub Wlodarczyk
# Note for future... Clean up majorly needed. Organize into classes.

import numpy as np
import cv2
import time
from PIL import ImageGrab, Image
from mss import mss
from pynput import keyboard

#bbox = (0, 0, 1520, 900)
bbox = (0, 175, 501, 400)

sct = mss()

key_dict = {
    keyboard.Key.left  : False,
    keyboard.Key.right : False,
    keyboard.Key.up    : False,
    keyboard.Key.down  : False
}

curr_output = None
training_data = []
file_name = "D:/Projects/SelfDrivingFZero/trainingData"
saved = False
paused = True

def build_output(key_dict):
    forward       = [1,0,0,0,0,0,0]
    back          = [0,1,0,0,0,0,0]
    left          = [0,0,1,0,0,0,0]
    right         = [0,0,0,1,0,0,0]
    forward_left  = [0,0,0,0,1,0,0]
    forward_right = [0,0,0,0,0,1,0]
    noop          = [0,0,0,0,0,0,1]

    key_up = keyboard.Key.up
    key_down = keyboard.Key.down
    key_left = keyboard.Key.left
    key_right = keyboard.Key.right

    if key_dict[key_up]:
        return forward
    if key_dict[key_left] and key_dict[key_up]:
        return forward_left
    if key_dict[key_right] and key_dict[key_up]:
        return forward_right
    if key_dict[key_right]:
        return right
    if key_dict[key_left]:
        return left
    if key_dict[key_down]:
        return back

    return noop
        
def on_press(key):
    global curr_output
    global paused
    global key_dict

    key_dict[key] = True
    curr_output = build_output(key_dict)

    if type(key) is not keyboard.Key and key.char == 'w':
        paused = not paused
        print("Paused" if paused else "Unpaused")

def on_release(key):
    global curr_output
    global key_dict
    key_dict[key] = False

    curr_output = build_output(key_dict)

lis = keyboard.Listener(on_press=on_press, on_release=on_release)
lis.start()

while True:
    before = time.time()

    img = sct.grab(bbox)
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 30, 120)
    cv2.imshow('window', gray)

    #Janky need a better way to do this.
    if(curr_output and not paused):
        training_data.append([gray, curr_output])

    if(len(training_data) % 500 == 0):
        print("Saved " + str(len(training_data)) + " sets of test data.")

    if(len(training_data) == 10000 and not saved):
        print("Saving data to " + file_name)
        np.save(file_name, training_data)
        saved = True

    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):
        cv2.destroyAllWindows()
        lis.stop()
        break
        
    after = time.time()


lis.join()