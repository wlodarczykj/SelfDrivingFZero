import numpy as np
import cv2
import time
from PIL import ImageGrab, Image
from mss import mss
from pynput import keyboard

#bbox = (0, 0, 1520, 900)
bbox = (0, 175, 501, 380)

sct = mss()

forward = [1,0,0,0]
back = [0,1,0,0]
left = [0,0,1,0]
right = [0,0,0,1]

output_dict = {
    keyboard.Key.left : left,
    keyboard.Key.right : right,
    keyboard.Key.up : forward,
    keyboard.Key.down : back
}
curr_output = None

def training_output(key):
    if(key in output_dict):
        return output_dict[key]
    else:
        return [0,0,0,0]

def on_press(key):
    try: k = key.char 
    except: k = key.name 
    global curr_output
    curr_output = training_output(key)
    # print('Key pressed: ' + k)

lis = keyboard.Listener(on_press=on_press)
lis.start()

training_data = []
file_name = "D:/Projects/SelfDrivingFZero/training.data"
saved = False

while True:
    before = time.time()

    img = sct.grab(bbox)
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    cv2.imshow('window', gray)

    #Janky need a better way to do this.
    if(curr_output):
        training_data.append([edges, curr_output])

    if(len(training_data) == 1 and not saved):
        np.save(file_name, training_data)
        saved = True

    curr_output = None

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        lis.stop()
        break
        
    after = time.time()

    print("inner loop took : " + str(round((after - before),4) * 1000) + " ms")

lis.join()