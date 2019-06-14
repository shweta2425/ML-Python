#!/usr/bin/env python
# coding: utf-8

# In[1]:


import theano
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import keras # Test out Theano when time permits as well
from PIL import Image


# In[2]:



# classifier = pickle.load(f)
# test_data_gen = pickle.load(f)


file = open('model.pkl', 'rb')
classifier=pickle.load(file)
test_datagen=pickle.load(file)


# In[3]:


Image.open('dataset/training_set/cats/cat.1.jpg').resize((256,256))


# In[6]:


#Prediction of image
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# img1 = image.load_img('dataset/training_set/cats/cat.1.jpg', target_size = (64, 64))
img1 = image.load_img('dataset/test_set/dogs/dog.4003.jpg', target_size = (64, 64))
# img1 = image.load_img('test/cats/dog.4003.jpg', target_size = (64, 64))
# img1 = image.load_img('test/Cat/10.jpg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
if(prediction[:,:]>0.5):
    value ='Dog :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Cat :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

plt.imshow(img1)
plt.show()


# In[ ]:




