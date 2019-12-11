#!/usr/bin/env python
# coding: utf-8
# %%
import os
import cv2
import numpy as np
import json
import sys
import pickle
from moviepy.editor import *
from IPython.display import HTML 
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image


# %%
origin_video_mode = 1

# initialize
path="../../data/train_data/"
model_name = '../../data/model/straight/51.951698461546045_Nov-26-2019_08-22-06.h5'
img_width, img_height, channel = 400, 400, 1



path_list=os.listdir(path)
json_dic = []
clips = []
# graphic
time_index = 1
time_series = []

def handle_img_pre(img):
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_mod = load_in_img(gray_image)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    left_dir, right_dir, left, right = handle_dir(classes[1][0][0], classes[3][0][0])
    cv2.putText(img, "left_wheel_speed: " + str(int(classes[0][0][0]) + 10), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "left_wheel_dir: " + str(left_dir) + '(' + str(left) + ')', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_speed: " + str(int(classes[2][0][0]) + 10), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_dir: " + str(right_dir) + '(' + str(right) + ')', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    return img


# %%

if (origin_video_mode):
    path_list.sort()
    for dirname in path_list:
        file_path = os.path.join(path, dirname)
        
        if(os.path.isdir(file_path)):
            print(file_path)
#             file_path = os.listdir(file_path)
            dirname=os.listdir(path + '/' + dirname)
            dirname.sort()
            for filename in dirname:
#                 print(filename)
                if (not filename.endswith('.png')):
                    continue
                image1 = cv2.imread(os.path.join(file_path,filename))
                cv2.imwrite('out.jpg', image1)
                clips.append(ImageClip('out.jpg').set_duration(0.05))
        else:
            print(file_path)
            image1 = cv2.imread(file_path)
            cv2.imwrite('out.jpg', image1)
            clips.append(ImageClip('out.jpg').set_duration(0.05))
            
    video = concatenate(clips, method="compose")
    video.write_videofile('origin.mp4', fps=15)


# %%
def load_in_img(image):
    image = cv2.resize(image, (img_width,img_height), fx=0, fy=0)
    image = image.reshape(img_width, img_height, channel)
    return image


# %%


from keras.preprocessing import image
def handle_dir(left, right):
#     return left, right
    if left < 0.65:
        return 0, 1, left, right
    return 1, 1, left, right

def handle_img(img):
    if(channel == 1):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_mod = load_in_img(gray_image)
    elif(channel == 3):
        image_mod = load_in_img(img)
    x = image.img_to_array(image_mod)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    left_dir, right_dir, left, right = handle_dir(classes[1][0][0], classes[3][0][0])
    cv2.putText(img, "left_wheel_speed: " + str(int(classes[0][0][0])), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "left_wheel_dir: " + str(left_dir) + '(' + str(left) + ')', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_speed: " + str(int(classes[2][0][0])), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_dir: " + str(right_dir) + '(' + str(right) + ')', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return img

    


# %%
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")
model.summary()

test_clip_filename = './origin.mp4'
new_clip_output = './predict.mp4'
test_clip = VideoFileClip(test_clip_filename)
new_clip = test_clip.fl_image(lambda x: handle_img(x)) #NOTE: this function expects color images!!
new_clip.write_videofile(new_clip_output,audio=False, fps = 4)


# %%


# %%




