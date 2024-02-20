#!/usr/bin/env python
# coding: utf-8

# In[32]:


pip install tensorflow


# In[31]:


pip install opencv-python


# In[2]:


import tensorflow as tf
mnist=tf.keras.datasets.mnist


# In[3]:


print(mnist)


# In[9]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0


# In[10]:


model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                 tf.keras.layers.Dense(128,activation ='relu'),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(10,activation='softmax'),])


# In[6]:


#1.) why to divide by 255
# when we are working with image data,the pixel values are integers in the 
#range of [0,255].so,dividig it by 255.0 scales these values to the range of
# [0,1]
# working with the smaller values increase the stability of optimization algorithm

#tf.keras.layers.dense(128, activation='relu'),
# 2.) why 128
#It is a specific number of neurons or units in the dense layer
#relu -->Received Linear Unit , it helps to add non-linearity to our algorithm


# In[12]:


model.compile(optimizer='adam', loss="sparse_categorical_crossentropy" ,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)


# 

# In[13]:


test_loss, test_accuracy=model.evaluate(x_test, y_test)


# In[14]:


print(test_loss)
print(test_accuracy)


# In[15]:


#Activation function
#Relu
#It is one of the most widely used activation function,It replaces all negative values with zero, leaving 
#positive values unchanged

#signoid 
#signoid reduces the output b/w 0 and 1 ,making it suitable for binary classification problems 

#Tanh (Hyperbolic Tangent)
#It reduces the b/w -1 and 1

#softmax
#It is mostly used in output layer

#leaku Relu
#It is a another varient of Relu that allows a small positive gradient for negative value just to avoid dead neurons



# In[16]:


#Project:1
#Convert image to greyscall using CNN

import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt 
import numpy as np


# In[17]:


#Load the RBG image
image_path ="C:\\Users\\swath\\OneDrive\\Pictures\\Mobile piks\\IMG-20220922-WA0030.jpg"
original_image=load_img(image_path,target_size =(224,224))
original_array =img_to_array(original_image)
original_array=original_array/225.0
print(original_array)


# In[ ]:





# In[18]:


plt.figure(figsize=(6,6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_array)


# In[19]:


#Convert image to grey scale 
model = models.Sequential()

#Sequentials() --> It allows us to create a linear stack of layers in a neural network.
#you can add one layer at neural metwork at a time and each layer has connection only to the prevoius and next layer


model.add(layers.Conv2D(1, (3, 3 ), activation='relu', input_shape=(224, 224, 3)))
#conv2d --> It represent 2d conventional layer

#conventional layer-->used to perform element value
#multiplication or addition or diviison etc..
#layers.conv2d(1) -->1 is used to mention the number of filters in the conventional layer 
#(3,3) -->set the size of filter 
#(224,224,3)-->224x224 -->pixels with 3 color channel(RGB)

model.add(layers.MaxPooling2D((2,2)))

#MaxPooling2D() -->it is used to set the dimensions of input data and extract important features from Conv2D layer

model.summary()

#To reshape the image

input_image=np.expand_dims(original_array,axis=0)
greyscale=model.predict(input_image)

plt.figure(figsize=(6,6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_array)


plt.figure(figsize=(6,6))
plt.subplot(1, 2, 2)
plt.title("Original Image")
plt.imshow(np.squeeze(greyscale),cmap='gray')


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
#model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical

from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
#from zipfile import Zipfile
from PIL import Image


# In[59]:


X=[]
Z=[]
IMG_SIZE=150

FLOWER_DAISY_DIR='C:\\Users\\swath\\OneDrive\\Desktop\\zipfolder\\archive (4)\\train\\daisy'
FLOWER_SUNFLOWER_DIR='C:\\Users\\swath\\OneDrive\\Desktop\\zipfolder\\archive (4)\\train\\dandelion'
FLOWER_TULIP_DIR='C:\\Users\\swath\\OneDrive\\Desktop\\zipfolder\\archive (4)\\train\\rose'
FLOWER_DANDI_DIR='C:\\Users\\swath\\OneDrive\\Desktop\\zipfolder\\archive (4)\\train\\sunflower'
FLOWER_ROSE_DIR='C:\\Users\\swath\\OneDrive\\Desktop\\zipfolder\\archive (4)\\train\\tulip'


# In[60]:


def assign_label(img,flower_type):
  return flower_type


# In[61]:


#tqdm-->It creates a progress bar from the loop
def make_train_data(flower_type, DIR):
  for img in tqdm(os.listdir(DIR)):
    label=assign_label(img, flower_type)
    path=os.path.join(DIR, img)
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    img=cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(np.array(img))
    Z.append(str(label))


# In[62]:


make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[63]:


make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[64]:


make_train_data('Tulip', FLOWER_TULIP_DIR)
print(len(X))


# In[65]:


make_train_data('Dandelion', FLOWER_DANDI_DIR)
print(len(X))


# In[66]:


make_train_data('Rose', FLOWER_ROSE_DIR)
print(len(X))


# In[67]:


fig, ax=plt.subplots(5,2)
fig.set_size_inches(15, 15)

for row in range(5):
  for col in range(2):
    l=rn.randint(0, len(X))
    ax[row, col].imshow(X[l])
    ax[row, col].set_title("Flower: "+Z[l] )

    plt.tight_layout()


# In[ ]:





# In[ ]:




