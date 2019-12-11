#!/usr/bin/env python
# coding: utf-8
# + {}
#configure path
csv_file_name = '../../data/csv/straight_store.csv'
model_path = '../../data/model/straight/'
INPUT_SHAPE = (400, 400, 3) # height width

#assign the specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -

# 自動增長 GPU 記憶體用量
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config = tf.ConfigProto(allow_soft_placement=True)
session = tf.Session(config=config)
ktf.set_session(session)
graph = False

# load in the data
import pandas as pd
import cv2
import numpy as np

dir_log = pd.read_csv(csv_file_name,usecols=[1,3])
speed_log = pd.read_csv(csv_file_name,usecols=[2,4])
filename_list= pd.read_csv(csv_file_name,usecols=[5])

dir_log = np.array(dir_log).tolist()
speed_log = np.array(speed_log).tolist()
filename_list = np.array(filename_list).tolist()

## data augmentation
turn_dir_list = []
turn_speed_list = []
turn_list_img = []
straight_dir_list = []
straight_speed_list = []
straight_list_img = []
for i in range(len(dir_log)):
    if dir_log[i][0] == 0:
        turn_dir_list.append(dir_log[i])
        turn_speed_list.append(speed_log[i])
        turn_list_img.append(filename_list[i])
    else:
        straight_dir_list.append(dir_log[i])
        straight_speed_list.append(speed_log[i])
        straight_list_img.append(filename_list[i])

turn_dir_list_ = []
turn_speed_list_= []
turn_list_img_=[]
## extend turn data
while(len(turn_dir_list_) < len(straight_dir_list)):
    turn_dir_list_.extend(turn_dir_list)
    turn_speed_list_.extend(turn_speed_list)
    turn_list_img_.extend(turn_list_img)

dir_log = []
speed_log = []
filename_list = []

dir_log.extend(turn_dir_list_)
dir_log.extend(straight_dir_list)
speed_log.extend(turn_speed_list_)
speed_log.extend(straight_speed_list)
filename_list.extend(turn_list_img_)
filename_list.extend(straight_list_img)

track_log = []
for i in range(len(dir_log)):
    data_frame = []
    data_frame.extend(speed_log[i])
    data_frame.extend(dir_log[i])
    data_frame.extend(filename_list[i])
    track_log.append(data_frame)


track_log=pd.DataFrame(track_log)
track_log.rename(columns={0:'left_wheel_speed',1:'right_wheel_speed',2:'left_wheel_dir',3:'right_wheel_dir',4:'filename'},inplace=True)

def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    if(INPUT_SHAPE[2] == 1):
        image = cv2.imread(imageLocation, 0) # Gray
    elif(INPUT_SHAPE[2] == 3):
        image = cv2.imread(imageLocation)    # BGR

    if (image is None):
        print(imageLocation)
        
    image = cv2.resize(image, (INPUT_SHAPE[0],INPUT_SHAPE[1]), fx=0, fy=0)
    image = image.reshape(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    return image

numSample = 50
centerImgs = np.array([load_in_img(img_location) for img_location in track_log['filename'][0:numSample]], dtype=np.float32)

import cv2
import numpy as np

## random shift image
def random_shift_image(image):
    dx = int(160 * (np.random.rand()-0.5))
    image = np.roll(image, dx, axis=1)
    if dx>0:
        image[:, :dx] = 1
    elif dx<0:
        image[:, dx:] = 1
        
    shift_speed = int(dx / 8)

    return (image, shift_speed)

## process image information
def process_image(img, left_speed, left_dir, right_speed, right_dir):
    img, shift_speed = random_shift_image(img)
    if left_speed > 0:
        left_speed = left_speed + shift_speed
    if left_speed < 0:
        left_speed = 0
    return img, left_speed, left_dir, right_speed, right_dir

import sklearn

#load in the img and control data
def load_data(img_sample, left_speed, left_dir, right_speed, right_dir, correction = 10):

    img_center = load_in_img(img_sample)
    
    left_speed_center = float(left_speed) * 1.0
    
    right_speed_center = float(right_speed) * 1.0

    return (img_center,img_center,img_center), (left_speed_center, left_speed_center, left_speed_center), (left_dir, left_dir, left_dir), (right_speed_center, right_speed_center, right_speed_center), (right_dir, right_dir, right_dir)

#Generator
def generator(samples, batch_size=64, mode='default'):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            #print("in")
            batch_samples = samples[offset:offset + batch_size]   
            #print(batch_samples)
            images = []
            left_speeds = []
            left_dirs = []
            right_speeds = []
            right_dirs = []
            target = []
            for img, left_speed, left_dir, right_speed, right_dir in zip(batch_samples['filename'], batch_samples['left_wheel_speed'], batch_samples['left_wheel_dir'], batch_samples['right_wheel_speed'], batch_samples['right_wheel_dir']):
                image, left_speed, left_dir, right_speed, right_dir = load_data(img, left_speed, left_dir, right_speed, right_dir)
                
                
                for item in zip(image, left_speed, left_dir, right_speed, right_dir): #iterate camera images and steering angles
                    aug_image, augleft_speed, augleft_dir, augright_speed, augright_dir = process_image(item[0], item[1], item[2], item[3], item[4])
                    images.append(aug_image)
                    left_speeds.append(augleft_speed)
                    left_dirs.append(augleft_dir)
                    right_speeds.append(augright_speed)
                    right_dirs.append(augright_dir)
                    target.append([augleft_speed, augleft_dir, augright_speed, augright_dir])
                
            X_train = np.array(images)
            y_train = np.array([left_speeds, left_dirs, right_speeds, right_dirs])
            left_speeds = np.array(left_speeds)
            left_dirs = np.array(left_dirs)
            right_speeds = np.array(right_speeds)
            right_dirs = np.array(right_dirs)
            
            if(mode == 'speed'):
                yield X_train, [left_speeds, right_speeds]
            elif(mode == 'dir'):
                yield X_train, [left_dirs, right_dirs]
            else:
                yield X_train, [left_speeds, left_dirs, right_speeds, right_dirs]

# split the log into train_samples and validation_samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(track_log, test_size=0.2)

# see if generator work
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
if graph:
    iterator = generator(train_samples, 64)
    sample_images, target = iterator.__next__()

    plt.subplots(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.title("speed: {:.4f}".format(target[0][i+20]))
        show_image = sample_images[i].reshape(200, 100)
        plt.imshow(show_image)
    plt.show()

# AGV's model
import pickle
import numpy as np
import math
from keras.utils import np_utils
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Activation, Concatenate
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.models import Model
from keras import backend as K

losses = {
    "right_dir_output": "binary_crossentropy",
    "left_dir_output": "binary_crossentropy",
    "left_speed_output": "mse",
    "right_speed_output": "mse",
}
DROP_PROB = 0.7
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# +
main_input = Input(shape=INPUT_SHAPE, name='main_input')
x = Conv2D(24, (3, 3), activation='relu', strides=(2, 2))(main_input)
x = Conv2D(24, (3, 3), activation='relu', strides=(2, 2))(x)
x = Conv2D(36, (5, 5), activation='relu', strides=(2, 2))(x)
x = Conv2D(48, (5, 5), activation='relu', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
x = Dropout(DROP_PROB)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Dense(128)(x)
x = Dense(10, name='speed')(x)

left_speed_output = Dense(1, name='left_speed_output')(x)
left_dir_output_ = Dense(5, name='left_dir_output_')(left_speed_output)

left_concat_layer= Concatenate()([left_dir_output_, x])
left_dir_output = Dense(1, name='left_dir_output', activation='sigmoid')(left_concat_layer)

right_speed_output = Dense(1, name='right_speed_output')(x)
right_dir_output_ = Dense(5,name='right_dir_output_')(right_speed_output)

right_concat_layer= Concatenate()([right_dir_output_, x])
right_dir_output = Dense(1,name='right_dir_output', activation='sigmoid')(right_concat_layer)
model = Model(inputs=[main_input], outputs=[left_speed_output, left_dir_output, right_speed_output, right_dir_output])
model.summary()
model.compile(optimizer="adam", loss=losses)
# model summary 
# -

import time, os, fnmatch, shutil
def save_model(model_name):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    model_name = (model_path + model_name  + '_' + timestamp + '.h5')
    
    return model_name


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
import keras

# compile and train the model using the generator function
nb_epoch_count = 30
for index in range(40):
    train_generator = generator(train_samples, 64)
    validation_generator = generator(validation_samples, 4)
    
    history_object = model.fit_generator(
                                      train_generator, 
                                      steps_per_epoch=200, 
                                      validation_data=validation_generator, 
                                      validation_steps=len(validation_samples)/2, 
                                      epochs=nb_epoch_count, 
                                      verbose=1)
    
    print('history_object')
    print(history_object)
    h5_output = save_model(str(history_object.history['loss'][nb_epoch_count-1]) + str(index)) 
    model.save(h5_output)
    print("Model saved")


    print('Time ',index+1)

# summarize history for loss
if graph:
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()




