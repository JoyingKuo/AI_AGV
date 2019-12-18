#!/usr/bin/env python
# coding: utf-8
# + {}
#!/usr/bin/env python
# coding: utf-8
# # + {}

import os, cv2, time, pickle, fnmatch,math, shutil
import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Activation, Concatenate
import keras.backend.tensorflow_backend as ktf
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback

# +
## 環境變數設置
# 指定要使用哪一片GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# 自動增長 GPU 記憶體用量
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config = tf.ConfigProto(allow_soft_placement=True)
session = tf.Session(config=config)
ktf.set_session(session)
graph = False 

## 路徑初始化
loaded_csv_path = '../../data/csv/straight_store.csv'
model_save_path = '../../data/model/straight/'
dataset = [] # 存處理過後的訓練資料


##參數初始化
# 要訓練的圖片大小以及channel(rgb, gray...)
image_width = 400 
image_height = 400 
image_channel = 1

# 驗證資料佔所有資料的百分比
validation_size = 0.2
# 幾個epoch存一次model
save_model_in_number_of_epoch = 30
# 
Steps_per_epoch = 200 # steps_per_epoch* batch_size = number of total image


# +
# 因為csv中的資料尚未平衡過(通常直行的資料量 >>> 轉彎)
# 需要平衡資料, 將兩類資料平衡, 達到更好的訓練效果

def data_balance_and_augmention():
    global dataset
    #從csv讀取各行資料
    dir_log = pd.read_csv(loaded_csv_path,usecols=[1,3]) # 左方向, 右方向
    speed_log = pd.read_csv(loaded_csv_path,usecols=[2,4]) # 左輪速, 右輪速
    filename_list= pd.read_csv(loaded_csv_path,usecols=[5]) # 檔案名稱

    # 將資料格式轉為list
    dir_log = np.array(dir_log).tolist()
    speed_log = np.array(speed_log).tolist()
    filename_list = np.array(filename_list).tolist()

    # 將轉彎與直行的資料分開存
    turn_dir_list = [] 
    turn_speed_list = []
    turn_list_img = []
    straight_dir_list = []
    straight_speed_list = []
    straight_list_img = []

    # 遍歷所有資料
    for i in range(len(dir_log)):
        if dir_log[i][0] == 0: #左輪方向為0, 表示轉彎
            turn_dir_list.append(dir_log[i])
            turn_speed_list.append(speed_log[i])
            turn_list_img.append(filename_list[i])
        else:
            straight_dir_list.append(dir_log[i])
            straight_speed_list.append(speed_log[i])
            straight_list_img.append(filename_list[i])

    # 進行資料擴增
    turn_dir_list_ = []
    turn_speed_list_= []
    turn_list_img_=[]
    # 如果轉彎的資料量 < 直行的資料,就將轉彎資料擴增一倍
    while(len(turn_dir_list_) < len(straight_dir_list)):
        turn_dir_list_.extend(turn_dir_list)
        turn_speed_list_.extend(turn_speed_list)
        turn_list_img_.extend(turn_list_img)


    # 再將轉彎資料與直行資料接起來
    dir_log = []
    speed_log = []
    filename_list = []

    dir_log.extend(turn_dir_list_)
    dir_log.extend(straight_dir_list)
    speed_log.extend(turn_speed_list_)
    speed_log.extend(straight_speed_list)
    filename_list.extend(turn_list_img_)
    filename_list.extend(straight_list_img)

    # 將圖片資料一筆一筆接起來
    for i in range(len(dir_log)):
        data_frame = [] 
        data_frame.extend(speed_log[i])
        data_frame.extend(dir_log[i])
        data_frame.extend(filename_list[i])
        dataset.append(data_frame)


    dataset=pd.DataFrame(dataset)
    dataset.rename(columns={0:'left_wheel_speed',1:'right_wheel_speed',2:'left_wheel_dir',3:'right_wheel_dir',4:'filename'},inplace=True)

    return


# -

def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    if(image_channel == 1):
        image = cv2.imread(imageLocation, 0) # Gray
    elif(image_channel == 3):
        image = cv2.imread(imageLocation)    # BGR

    if (image is None):
        print(imageLocation)
        
    image = cv2.resize(image, (image_width,image_height), fx=0, fy=0)
    image = image.reshape(image_width, image_height, image_channel)
    return image

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
            for image, left_speed, left_dir, right_speed, right_dir in zip(batch_samples['filename'], batch_samples['left_wheel_speed'], batch_samples['left_wheel_dir'], batch_samples['right_wheel_speed'], batch_samples['right_wheel_dir']):
                images.append(load_in_img(image))
                left_speeds.append(left_speed*1.0)
                left_dirs.append(left_dir)
                right_speeds.append(right_speed*1.0)
                right_dirs.append(right_dir)
                
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


def save_model(model_name):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    model_name = (model_path + model_name  + '_' + timestamp + '.h5')
    
    return model_name


def define_training_network():
    # activation function
    losses = {
        "right_dir_output": "binary_crossentropy",
        "left_dir_output": "binary_crossentropy",
        "left_speed_output": "mse",
        "right_speed_output": "mse",
    }
    
    DROP_PROB = 0.7 #隨機丟掉神經元,可以避免overfitting
    main_input = Input(shape=(image_width, image_height,image_channel), name='main_input')
    x = Conv2D(24, (3, 3), activation='relu', strides=(2, 2))(main_input)
    x = Conv2D(24, (3, 3), activation='relu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='relu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    x = Dropout(DROP_PROB)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(128)(x)
    x = Dense(10)(x)

    left_speed_output = Dense(1, name='left_speed_output')(x)
    left_dir_input = Dense(5, name='left_dir_input')(left_speed_output)

    left_concat_layer= Concatenate()([left_dir_input, x])
    left_dir_output = Dense(1, name='left_dir_output', activation='sigmoid')(left_concat_layer)

    right_speed_output = Dense(1, name='right_speed_output')(x)
    right_dir_input = Dense(5,name='right_dir_input')(right_speed_output)

    right_concat_layer= Concatenate()([right_dir_input, x])
    right_dir_output = Dense(1,name='right_dir_output', activation='sigmoid')(right_concat_layer)
    
    model = Model(inputs=[main_input], outputs=[left_speed_output, left_dir_output, right_speed_output, right_dir_output])
    model.summary()
    model.compile(optimizer="adam", loss=losses)
    # model summary 

    return model


def train(model):
    # 將資料分成測試資料及驗證資料
    train_samples, validation_samples = train_test_split(dataset, test_size= validation_size)

    for index in range(40):
        train_generator = generator(train_samples, 64)
        validation_generator = generator(validation_samples, 4)

        history_object = model.fit_generator(
                                          train_generator, 
                                          steps_per_epoch=Steps_per_epoch, 
                                          validation_data=validation_generator, 
                                          validation_steps=len(validation_samples)/2, 
                                          epochs=save_model_in_number_of_epoch, 
                                          verbose=1)

        print('history_object')
        print(history_object)
        h5_output = save_model(str(history_object.history['loss'][save_model_in_number_of_epoch-1]) + str(index)) 
        model.save(h5_output)
        print("Model saved")

        print('Time ',index+1)



# +
def main():
    data_balance_and_augmention()
    model = define_training_network()
    train(model)

if __name__ == '__main__':
    main()
# -


