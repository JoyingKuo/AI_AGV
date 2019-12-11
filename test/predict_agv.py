#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image

img_width, img_height = 400, 400

def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    image = cv2.imread(imageLocation) # Gray

    if (image is None):
        print(imageLocation)
        
    image = image[45:-9,::]
    image = cv2.resize(image, (400,400), fx=0, fy=0)
    image = image.reshape(400, 400, 1)
    return image


from keras.preprocessing import image
def handle_dir(left, right):
#     return left, right
    if left < 0.65:
        return 0, 1, left, right
    return 1, 1, left, right

def handle_img():
    image_mod = load_in_img('./img.png')
    x = image.img_to_array(image_mod)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    left_dir, right_dir, left, right = handle_dir(classes[1][0][0], classes[3][0][0])
    

    if (int(classes[0][0][0]))<= 0.0 :
        speed_left = 0
    else:
        speed_left = int(classes[0][0][0])

    if (int(classes[2][0][0]))<= 0.0 :
        speed_right = 0
    else:
        speed_right = int(classes[2][0][0])
    
    command_string = 'mv value_*.txt value_' + str(left_dir) + '_' + str(right_dir) + '_' + str(speed_left) + '_' + str(speed_right) + '.txt'
    print(command_string)
    os.system('mv value_*.txt value_0_0_0_0.txt')
    os.system(command_string)
    os.system('mv status_1.txt status_2.txt')

    


#     cv2.putText(img, "left_wheel_speed: " + str(int(classes[0][0][0]) + 10), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(img, "left_wheel_dir: " + str(left_dir) + '(' + str(left) + ')', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(img, "right_wheel_speed: " + str(int(classes[2][0][0]) + 10), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(img, "right_wheel_dir: " + str(right_dir) + '(' + str(right) + ')', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)



model_name = '../../data/model/model_dir_ver4/12.7279729479321557_Nov-13-2019_10-08-49.h5'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")
os.system('rm value*.txt')
os.system('touch value_0_0_0_0.txt')
filepath = "./status_1.txt"


try:
    while(1):
        if os.path.isfile(filepath):
            handle_img()
except KeyboardInterrupt:
    print ('Exception: KeyboardInterrupt')


