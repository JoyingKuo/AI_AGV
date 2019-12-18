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

## 環境變數設置
os.environ['HDF5_USE_FILE_LOCKING']='FALSE' #為了避免 h5py default 的 file lock, 存model的時候才不會被拒絕 

## 路徑初始化
loaded_model_path = '../../data/model/model_dir_ver4/12.7279729479321557_Nov-13-2019_10-08-49.h5' # 要被載入的 model 路徑
image_path = '../../data/108-10-16/test -4/2019-10-16-15-16-49-1184-p078-p080.png' # 要被評估的照片路徑


##參數初始化

# test image 要被resize 到跟training image 一樣的大小才可以被 predict
image_reshape_width = 400
image_reshape_height = 400 
image_channel = 1

# 輸出模式, 0:代表預測單張圖片, 1:代表預測整個資料夾圖片並將結果輸出成影片
output_mode = 0 

# output mode = 1 時需要設定影片參數
test_data_path='../../data/train_data/' # 訓練資料路徑
origin_clip_name = './origin.mp4' #原始照片所合成的影片名稱
predict_clip_name = './predict.mp4' #要預測的圖片所合成的影片名稱
FPS = 4 # frame per second
duration = 0.05 # 一張照片的播放時間



def image_preprocessing(img):

    if(image_channel == 1):
        # 轉換色相至灰階(1 channel), 才可以進行後續的reshpe
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # reshpae image 至 (400, 400) 
    img = cv2.resize(img, (image_reshape_width, image_reshape_height), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(image_reshape_width, image_reshape_height, image_channel)

    #  轉換image 至可以被餵進 model 的資料型態
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.vstack([img_array])

    return img, img_array

def generate_origin_clip(clip_name):
    clip_list = [] # 每個元素都是 single image clip
    
    # 在訓練資料路徑的子資料夾列表
    sub_dir_list=os.listdir(train_data_path) 
    sub_dir_list.sort() 

    # 依序讀取每個訓練資料夾
    for subdir in sub_dir_list:
        full_subdir_path = os.path.join(train_data_path, subdir) # 完整的subdir路徑

        # full_subdir_path 的資料型態必須為directory, 才可以繼續進行資料preprocess
        if(os.path.isdir(full_subdir_path)):

            # sorting資料夾裡的照片, 使其為正確排序
            img_path_list = os.listdir(full_subdir_path)
            img_path_list.sort()


            # 讀取資料夾裡的每張照片

            for img in img_path_list:
                img = cv2.imread(os.path.join(full_subdir_path,img))
                clip_list.append(ImageClip(img).set_duration(duration))

    # 將最後結果輸出成影片
    clip = concatenate(clip_list, method="compose")
    clip.write_videofile(origin_clip_name, fps=FPS)

    return

def generate_predict_clip(origin_clip_name):
    origin_clip = VideoFileClip(origin_clip_name)
    output_clip = origin_clip.fl_image(lambda x: predict_single_image(x)) #NOTE: this function expects color images!!
    output_clip.write_videofile(predict_clip_name,audio=False, fps = FPS)

    return


# 利用訓練好的model, 輸入照片並檢視其網路輸出結果
def predict_single_image(img)
    
    img, img_array = image_preprocessing(img)

    # 預測結果
    result = model.predict(img_array, batch_size=1)

    # 將預測出來的結果畫到圖片上
    left_dir, right_dir, left, right = handle_dir(result[1][0][0], result[3][0][0])
    cv2.putText(img, "left_wheel_speed: " + str(int(result[0][0][0])), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "left_wheel_dir: " + str(left_dir) + '(' + str(left) + ')', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_speed: " + str(int(result[2][0][0])), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_dir: " + str(right_dir) + '(' + str(right) + ')', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    

    return img

def main():
    # 載入先前訓練好的 model
    model = load_model(loaded_model)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    ##評估input image(s)的結果
    #評估整個資料夾並輸出影片
    if (output_mode):
        generate_origin_clip(origin_clip_name)
        generate_predict_clip(origin_clip_name)

    #評估單張圖片
    else:
        img = cv2.imread(image_path) 
        result = predict_single_image(img)

        # 顯示結果
        cv2.imshow(result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

if __name__ == '__main__':
    main()

