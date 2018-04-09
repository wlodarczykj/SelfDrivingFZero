import numpy as np
import cv2
import time
from PIL import ImageGrab, Image
from mss import mss
from pynput import keyboard

#bbox = (0, 0, 1520, 900)
bbox = (0, 175, 501, 380)

sct = mss()

def on_press(key):
    try: k = key.char 
    except: k = key.name 
    if key == keyboard.Key.esc: return False 
    print('Key pressed: ' + k)

lis = keyboard.Listener(on_press=on_press)
lis.start()

while True:
    before = time.time()

    img = sct.grab(bbox)
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    cv2.imshow('window', edges)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    after = time.time()

    # print("inner loop took : " + str(round((after - before),4) * 1000) + " ms")
lis.join()