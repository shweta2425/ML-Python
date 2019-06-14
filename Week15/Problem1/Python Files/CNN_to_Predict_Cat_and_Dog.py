#!/usr/bin/env python
# coding: utf-8

# In[1]:


import theano
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


# In[2]:


import keras # Test out Theano when time permits as well
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten


# In[3]:


# Initializing the ANN
classifier = Sequential()


# In[4]:


# step1 - Convolution layer

classifier.add(Convolution2D(32,3,3, input_shape =(64,64,3),activation='relu'))


# In[5]:


# step 2 - Pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[6]:



# classifier.add(Convolution2D(32,3,3,activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[7]:


# step 3 - Flattening

classifier.add(Flatten())


# In[8]:


# step 4 - Full connection

classifier.add(Dense(output_dim = 300, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))


# In[9]:


# step - 5 Compiling the CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


# fitting the CNN to our dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=80,
        epochs=25,
        validation_data=test_set,
        validation_steps=20)


# In[12]:


# save model
file = open('model.pkl', 'wb')
pickle.dump(classifier,file)
pickle.dump(test_datagen,file)
file.close()


# In[ ]:




