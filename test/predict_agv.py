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

## 環境變數設置
os.environ['HDF5_USE_FILE_LOCKING']='FALSE' #為了避免 h5py default 的 file lock, 存model的時候才不會被拒絕 

## 路徑初始化
loaded_model_path = '../../data/model/model_dir_ver4/12.7279729479321557_Nov-13-2019_10-08-49.h5'

##參數初始化
# test image 要被resize 到跟training image 一樣的大小才可以被 predict
img_width = 400
img_height = 400
img_channel = 1


def image_preprocessing(img):

    if(img_channel == 1):
        # 轉換色相至灰階(1 channel), 才可以進行後續的reshpe
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # reshpae image 至 (400, 400) 
    img = cv2.resize(img, (img_width, img_height),fx=0, fy=0)
    img = img.reshape(img_width, img_height, img_channel)

    #  轉換image 至可以被餵進 model 的資料型態
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.vstack([img_array])

    return img_array


def handle_direction(left, right):
    # 當得到的方向預測值 < 0.5, 就當作是 0(要轉彎), 否則為 1(不用轉彎)
    if left < 0.5:
        return 0, 1, left, right
    return 1, 1, left, right

def handle_img():
    #讀取image
    img = cv2.imread('./img.png')
    img_array = image_preprocessing(img)

    #預測結果
    result = model.predict(img_array, batch_size=1)
    left_dir, right_dir, left, right = handle_direction(result[1][0][0], result[3][0][0])
    
    #字串處理與寫檔
    if (int(result[0][0][0]))<= 0.0 :
        speed_left = 0
    else:
        speed_left = int(result[0][0][0])

    if (int(result[2][0][0]))<= 0.0 :
        speed_right = 0
    else:
        speed_right = int(result[2][0][0])
    
    command_string = 'mv value_*.txt value_' + str(left_dir) + '_' + str(right_dir) + '_' + str(speed_left) + '_' + str(speed_right) + '.txt'
    os.system('mv value_*.txt value_0_0_0_0.txt')
    os.system(command_string)
    os.system('mv status_1.txt status_2.txt')

    


# 載入model
model = load_model(loaded_model_path)
model.compile(optimizer="adam", loss="mse")

# 系統環境指令, 透過檔案跟agv溝通
os.system('rm value*.txt')
os.system('touch value_0_0_0_0.txt')
filepath = "./status_1.txt"


try:
    while(1):
        if os.path.isfile(filepath):
            handle_img()
except KeyboardInterrupt:
    print ('Exception: KeyboardInterrupt')


