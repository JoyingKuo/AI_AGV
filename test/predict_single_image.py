# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
import sys

# 環境變數設置
os.environ['HDF5_USE_FILE_LOCKING']='FALSE' #為了避免 h5py default 的 file lock, 存model的時候才不會被拒絕 

# 路徑初始化
loaded_model_path = '../../data/model/model_dir_ver4/12.7279729479321557_Nov-13-2019_10-08-49.h5' # 要被載入的 model 路徑
image_path = '../../data/108-10-16/test -4/2019-10-16-15-16-49-1184-p078-p080.png' # 要被評估的照片路徑

#參數初始化

# test image 要被resize 到跟training image 一樣的大小才可以被 predict
image_reshape_width = 400
image_reshape_height = 400  


def load_in_img(image_path):
    
    # 根據 image_path 讀取照片, 將其resize成(400,400)
    # 轉換色相至灰階並 reshape 其至 1 channel
    img = cv2.imread(image_path) 
    img = cv2.resize(img, (image_reshape_width, image_reshape_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape(image_reshape_width, image_reshape_height, 1)

    #  轉換image 至可以被餵進 model 的資料型態
    img = image.img_to_array(img) 
    img = np.expand_dims(img, axis=0) 
    img = np.vstack([img])

    return img


# 利用訓練好的model, 輸入照片並檢視其網路輸出結果
def predict(model,image_path)
    
    # 載入照片並做資料前處理
    img = load_in_img(image_path)

    # 預測結果
    result = model.predict(img, batch_size=10)

    return result

def main():
    # 載入先前訓練好的 model
    model = load_model(loaded_model)
    model.compile(optimizer="adam", loss="mse")

    #評估input image的結果
    result = predict(model, image_path)
    print ('result: ')
    print (result)

    

if __name__ == '__main__':
    main()

