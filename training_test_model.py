#訓練
#資料處理
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import zipfile
import PIL.Image as Image
data = np.array(data)

#根據mrl eye dataset訓練
local_zip = '/content/drive/MyDrive/archive.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

data = []
target = []
base_dir = "/content/mrleyedataset/"
categories = ['Close-Eyes', 'Open-Eyes']
for i in range(2):
    thedir = base_dir + categories[i]
    file_names = os.listdir(thedir)
    for fname in file_names[:320]:
        img_path = thedir + '/' + fname
        img = load_img(img_path , target_size = (224,224))
        x = img_to_array(img)
        data.append(x)
        target.append(i)

data = np.array(data)
x_train = preprocess_input(data)
N = len(categories)
y_train = to_categorical(target, N)

resnet = ResNet50V2(include_top=False, pooling="avg")
model = Sequential()
model.add(resnet)
model.add(Dense(N, activation='softmax'))
resnet.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)