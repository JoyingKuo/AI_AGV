#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import json
import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd


# 路徑初始化
train_data_path='../../data/train_data/' # 訓練資料路徑
csv_save_path='../../data/csv/' # 訓練資料列表的存放路徑
csv_file_name = 'train_list.csv'# 訓練資料列表的名稱
img_info_list = [] # 存訓練資料列表

# 參數初始化
black_scene_data_num = 200 #會產生200筆黑畫面(black.png)的資料至訓練資料列表
black_scene_img_path = train_data_path + 'black.png' # 黑畫面的圖片路徑

# 處理方向資料到可訓練的型態, p(正轉) => 1, n(反轉) => 0
def change_dir_str_to_int(direction):
    if direction == 'p':
        return 1
    else:
        return 0

# 處理黑畫面資料
def process_black_scene_data():
    for i in range(black_scene_data_num):
        single_img_info = {}
        # 強制 assign 黑畫面的輪子資訊為-左輪方向(1),左輪速(0),右輪方向(0),右輪速(0) 
        single_img_info['left_wheel_dir'] = 1
        single_img_info['left_wheel_speed'] = 0
        single_img_info['right_wheel_dir'] = 0
        single_img_info['right_wheel_speed'] = 0
        single_img_info['filename'] = black_scene_img_path
    
        img_info_list.append(single_img_info)
    return

# 處理一般畫面資料
def process_normal_scene_data():
    # 在訓練資料路徑的子資料夾列表
    sub_dir_list=os.listdir(train_data_path) 
    sub_dir_list.sort() 

    # 依序讀取每個訓練資料夾
    for subdir in sub_dir_list:
        print('Process directory : ' + subdir)

        full_subdir_path = os.path.join(train_data_path, subdir) # 完整的subdir路徑

        # full_subdir_path 的資料型態必須為directory, 才可以繼續進行資料preprocess
        if(os.path.isdir(full_subdir_path)):

            # sorting資料夾裡的照片, 使其為正確排序
            img_path_list = os.listdir(full_subdir_path)
            img_path_list.sort()
            
            
            # 讀取資料夾裡的每張照片
            for img in img_path_list:

                # 讀取照片檔名, 分析AGV的行走資訊, 左輪速/方向, 右輪速/方向
                x = img.split("-", 9)
                left_wheel = x[7]
                right_wheel = x[8].split(".", 1)[0]
                left_wheel_dir = change_dir_str_to_int(left_wheel[0:1])
                left_wheel_speed = left_wheel[1:]
                right_wheel_dir = change_dir_str_to_int(right_wheel[0:1])
                right_wheel_speed = right_wheel[1:]

                #將處理完的資訊存到list
                single_img_info = {}
                single_img_info['left_wheel_dir'] = left_wheel_dir
                single_img_info['left_wheel_speed'] = left_wheel_speed
                single_img_info['right_wheel_dir'] = right_wheel_dir
                single_img_info['right_wheel_speed'] = right_wheel_speed
                single_img_info['filename'] = full_subdir_path + '/' + img

                img_info_list.append(single_img_info)
    return

# 將 list 處理成 dataframe 形式, 以存成CSV
def process_list_data_to_dataframe():

    left_wheel_dir = [] # 只有存所有image的左輪方向資訊
    left_wheel_speed = [] # 只有存所有image的左輪速度資訊
    right_wheel_dir = [] # 只有存所有image的右輪方向資訊
    right_wheel_speed = [] # 只有存所有image的右輪速度資訊
    filename = [] # 只有存所有image的檔名資訊

    for data in img_info_list:
        left_wheel_dir.append(data['left_wheel_dir'])
        left_wheel_speed.append(data['left_wheel_speed'])
        right_wheel_dir.append(data['right_wheel_dir'])
        right_wheel_speed.append(data['right_wheel_speed'])
        filename.append(data['filename'])


    # 利用 pandas 這個套件, 可以將 list 直接轉換成 dataframe 
    df = pd.DataFrame.from_dict([])
    df['left_wheel_dir'] = left_wheel_dir
    df['left_wheel_speed'] = left_wheel_speed
    df['right_wheel_dir'] = right_wheel_dir
    df['right_wheel_speed'] = right_wheel_speed
    df['filename'] = filename

    # 儲存為CSV檔
    df.to_csv(csv_file_name, sep=',', encoding='utf-8')

    return


def main():

    # 檢查csv_save_path這個目錄是否已存在OS中, 沒有的話就創建這個資料夾
    # 若路徑不存在, 在寫入檔案到這個路徑的時候會造成檔案遺失
    if not os.path.isdir(csv_save_path):
        os.system('mkdir ' + csv_save_path)

    # 將 raw data 處理成 list 形式
    process_normal_scene_data();
    process_black_scene_data();
    
    # 將 list 處理成 dataframe 形式, 以存成CSV
    process_list_data_to_dataframe();


if __name__ == '__main__':
    main()
