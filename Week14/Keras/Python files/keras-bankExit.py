#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import math
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# loads and read csv file
df_original=pd.read_csv("Data/Churn_Modelling.csv",delimiter=",")
df =df_original
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


# checks data types of columns
df.dtypes


# In[6]:


# checks for null values
df.isnull().sum()


# In[7]:


# checks for duplicate values
df.duplicated().sum()


# In[8]:


# display column names
df.columns


# In[9]:


# display shape of the dataframe
df.shape


# In[10]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr,annot=True)


# In[11]:


# checks correlation with all columns
print(corr['Exited'].sort_values(ascending=True)[:])


# In[12]:


df['Exited'].value_counts()


# In[13]:


sb.countplot(x='Exited',data=df,palette='hls')


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


# split dataset into train and test
train,test = train_test_split(df,test_size=0.2)


# In[17]:


# saving test data into csv file
test.to_csv('test_file.csv',index=False,encoding='utf-8')


# In[18]:


# split train data into train and cross validation dataset
train_data,cross_val = train_test_split(train,test_size=0.2,random_state=0)


# In[19]:


# seperating features and labels from dataset
# Credit Score through Estimated Salary
x_train = train_data.iloc[:,3:13].values
# Exited
y_train = train_data.iloc[:,13].values


# In[20]:


x_train.shape,y_train.shape


# In[21]:



def categorical_encode(x):
   # Encoding categorical data country
    labelencoder_x_country = LabelEncoder()
    x[:,1] = labelencoder_x_country.fit_transform(x[:,1])
    # Encode categorical data gender
    labelencoder_x_gender = LabelEncoder()
    x[:,2]  =  labelencoder_x_gender.fit_transform(x[:,2])
    
    # Converting the string features into their own dimensions
    # Gender doesn't matter here because its binary
    OneHot_country = OneHotEncoder(categorical_features=[1]) # 1 is the country column
    x = OneHot_country.fit_transform(x).toarray()
    x = x[:, 1:]
    return x

# Encoding categorical (string based) data.
x_train = categorical_encode(x_train)


# In[22]:


# seperating features and labels from dataset of cross validation dataset
# Credit Score through Estimated Salary
x_cv = cross_val.iloc[:,3:13].values
# Exited
y_cv = cross_val.iloc[:,13].values


# In[23]:


# # Encoding categorical (string based) data.
x_cv = categorical_encode(x_cv)


# In[24]:


# Feature Scaling on independent variables
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_cv = sc.transform(x_cv)


# In[25]:


x_train.shape,y_train.shape,x_cv.shape,y_cv.shape


# In[26]:


import keras


# In[27]:


from keras.models import Sequential
from keras.layers import Dense


# In[28]:


# Initializing the ANN
classifier = Sequential()


# In[29]:


# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
classifier.add(Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))


# In[30]:


# Adding the second hidden layer
# Notice that we do not need to specify input dim. 
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 


# In[31]:


# Adding the output layer
# Notice that we do not need to specify input dim. 
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 


# In[32]:


# Compiling the Neural Network
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[33]:


# Fitting the Neural Network
classifier.fit(x_train, y_train, batch_size=10, epochs=100)


# # Predicting on the Cross Validation dataset result
# Testing the ANN

# In[34]:


y_pred = classifier.predict(x_cv)
print(y_pred)


# To use the confusion Matrix, we need to convert the probabilities that a customer will leave the bank into the form true or false.
# So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.

# In[35]:


y_pred = (y_pred > 0.5)
print(y_pred)


# In[36]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cv, y_pred)
print(cm)


# In[37]:


print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[38]:


# calculating accuracy
print (accuracy_score(y_cv,y_pred)*100,'% of cross validation data was classified correctly')


# In[39]:


# save model in pickle file
import pickle
fileobj=open('save_model.pkl','wb')
pickle.dump(classifier,fileobj)
pickle.dump(sc,fileobj)
fileobj.close()


# In[ ]:





# In[ ]:




