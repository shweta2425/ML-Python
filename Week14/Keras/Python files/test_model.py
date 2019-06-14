#!/usr/bin/env python
# coding: utf-8

# In[27]:


# import all libraries
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[28]:


# load test file
df = pd.read_csv('test_file.csv')
df.head()


# In[29]:


# load model
fileobj=open('save_model.pkl','rb')
classifier = pickle.load(fileobj)
sc = pickle.load(fileobj)


# In[30]:


# seperating features and labels from dataset
x_test = df.iloc[:,3:13].values
# Exited
y_test = df.iloc[:,13].values


# In[31]:


def categorical_encode(x):
   # Encoding categorical data country
    labelencoder_x_country = LabelEncoder()
    x[:,1] = labelencoder_x_country.fit_transform(x[:,1])
   # Encode categorical data gender
    labelencoder_x_gender = LabelEncoder()
    x[:,2]  =  labelencoder_x_gender.fit_transform(x[:,2])
    ohe_country = OneHotEncoder(categorical_features=[1])
    x = ohe_country.fit_transform(x).toarray()
    x = x[:, 1:]
    return x

x_test = categorical_encode(x_test)


# In[32]:


# transform test data 
x_test = sc.transform(x_test)


# In[33]:


# predict test result
prediction = classifier.predict(x_test)
prediction = (prediction > 0.5)


# In[34]:


# checking how many prediction are correct predicted and how many are wrongly predicted
cm = confusion_matrix(prediction,y_test)


# In[35]:


# calculating accuracy
print (accuracy_score(y_test,prediction)*100,'% of cross validation data was classified correctly')


# In[36]:


print (((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% of testing data was classified correctly')


# In[ ]:




