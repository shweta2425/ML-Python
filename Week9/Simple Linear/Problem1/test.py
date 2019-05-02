#!/usr/bin/env python
# coding: utf-8

# In[34]:


# import libraries
import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import pickle
import sklearn


# In[35]:


# read file
df_original=pd.read_csv("test_data.csv")

df =df_original
df.head()


# In[36]:


# separate labels and features
x_test=df.iloc[:,:-1].values
y_test=df.iloc[:,1].values
# print(y)


# In[37]:


# reading the pickle file

fileObject = open('train_data.pkl','rb')  
regressor = pickle.load(fileObject)  


# In[38]:


# getting the prediction values on train model
y_pred=regressor.predict(x_test)


# In[39]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df.head()


# In[40]:


# calculating accuracy on y_test data of test dataset
acc_test=sklearn.metrics.r2_score(y_test,y_pred)*100
print("Accuracy of test data =",acc_test)


# In[ ]:





# In[ ]:




