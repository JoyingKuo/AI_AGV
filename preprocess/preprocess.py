#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np
import json
import sys
import pickle
# from moviepy.editor import *
import matplotlib.pyplot as plt
import pandas as pd

# +
# initialize
path='../../data/train_data/'
csv_path='../../data/csv/'
csv_file_name = '../../data/csv/straight.csv'

path_list=os.listdir(path)
path_list.sort()
# -

# graphic
time_index = 1
time_series = []
json_dic = []


# 檢查目錄是否存在 
if not os.path.isdir(csv_path):
    os.system('mkdir ' + csv_path)

def define_dir(str):
    if str == 'p':
        return 1
    else:
        return 0

# for each dir in dir list
for dirname in path_list:
    print(dirname)
    file_path = os.path.join(path, dirname)
    
    if(os.path.isdir(file_path)):
        dirname=os.listdir(path + '/' + dirname)
        dirname.sort()
        
        # handle 降速
        down_speed_flag = 0
        
        # for each file in the test dir
        for filename in dirname:
            if(filename == 'black.png' ):
                continue
            one_info = {}
            time_series.append(time_index)
            time_index = time_index + 1

            x = filename.split("-", 9)
            left_wheel = x[7]
            right_wheel = x[8].split(".", 1)[0]
            image = cv2.imread(os.path.join(file_path,filename))
            left_wheel_dir = define_dir(left_wheel[0:1])
            left_wheel_speed = left_wheel[1:]
            right_wheel_dir = define_dir(right_wheel[0:1])
            right_wheel_speed = right_wheel[1:]
            if( right_wheel_speed == '0' ):
                continue

#             if( down_speed_flag == 0 and left_wheel_dir == 0 ):
#                 length = len(json_dic) - 1
#                 down_speed_flag = 1
#                 for i in range(15):
#                     json_dic[length - i]['left_wheel_speed'] = int(json_dic[length - i]['left_wheel_speed']) - 20
#                     json_dic[length - i]['right_wheel_speed'] = int(json_dic[length - i]['right_wheel_speed']) - 20
#             if( down_speed_flag == 1 and left_wheel_dir == 1 ):
#                 down_speed_flag = 0

            one_info['left_wheel_dir'] = left_wheel_dir
            one_info['left_wheel_speed'] = left_wheel_speed
            one_info['right_wheel_dir'] = right_wheel_dir
            one_info['right_wheel_speed'] = right_wheel_speed
            one_info['filename'] = file_path + '/' + filename

            # put text on image
#             cv2.putText(image, "left_wheel_dir: " + str(left_wheel_dir), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, "left_wheel_speed: " + str(left_wheel_speed), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, "right_wheel_dir: " + str(right_wheel_dir), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, "right_wheel_speed: " + str(right_wheel_speed), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            json_dic.append(one_info)

for i in range(200):
    one_info = {}
    one_info['left_wheel_dir'] = 1
    one_info['left_wheel_speed'] = 0
    one_info['right_wheel_dir'] = 0
    one_info['right_wheel_speed'] = 0
    one_info['filename'] = path + 'black.png'
    
    json_dic.append(one_info)

left_wheel_dir = []
left_wheel_speed = []
right_wheel_dir = []
right_wheel_speed = []
filename = []

for data in json_dic:
    left_wheel_dir.append(data['left_wheel_dir'])
    left_wheel_speed.append(data['left_wheel_speed'])
    right_wheel_dir.append(data['right_wheel_dir'])
    right_wheel_speed.append(data['right_wheel_speed'])
    filename.append(data['filename'])


# array to pd data type
df = pd.DataFrame.from_dict([])

df['left_wheel_dir'] = left_wheel_dir
df['left_wheel_speed'] = left_wheel_speed
df['right_wheel_dir'] = right_wheel_dir
df['right_wheel_speed'] = right_wheel_speed
df['filename'] = filename

df.to_csv(csv_file_name, sep=',', encoding='utf-8')


