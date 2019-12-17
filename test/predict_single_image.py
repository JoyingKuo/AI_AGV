# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
import sys

# configure path
model_name = '../../data/model/model_dir_ver4/12.7279729479321557_Nov-13-2019_10-08-49.h5'
image_path = '../../data/108-10-16/test -4/2019-10-16-15-16-49-1184-p078-p080.png'
def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    image = cv2.imread(imageLocation) # Gray

    if (image is None):
        print(imageLocation)
#     print(imageLocation)

#     image = image[45:-9,::]
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
    #image = color_frame_process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(400, 400, 1)
    return image

# dimensions of our images
img_width, img_height = 320, 240
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# +
# load the model we saved

print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")
# -

# predicting images
# img = load_in_img('./dir/test - 9/black.png')#black
# img = load_in_img('./dir/test - 改-第二個轉彎後繼續前進/2019-06-13-15-15-44-5910-p059-p058.png')#black
img = load_in_img(image_path)#second turn

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print ('op classes: ')
print (classes)



