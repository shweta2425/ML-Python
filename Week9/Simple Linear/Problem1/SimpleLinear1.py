#!/usr/bin/env python
# coding: utf-8

# In[57]:


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


# In[58]:


# read file
df_original=pd.read_csv("Salary_Data.csv")

df =df_original
df.head()


# In[59]:


# splitting data into train & test dataset
train,test=train_test_split(df,test_size=0.3,random_state=0)


# In[60]:


train.shape


# In[61]:


test.shape


# In[62]:


# saving datasets into csv filesS
test.to_csv('test_data.csv',index=False,encoding='utf-8')
train.to_csv('train_data.csv',index=False,encoding='utf-8')


# In[63]:


# loading training data csv file
train_df = pd.read_csv('train_data.csv')
train_df.head()


# In[64]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=train_test_split(train_df,test_size=0.3,random_state=0)


# In[65]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,1].values


# In[66]:


print("x_train",x_train.shape)
# print("x_test",x_cv.shape)
print("y_train",y_train.shape)
# print("y_test",y_cv.shape)


# In[67]:


# saving cross validation data into csv file
cv_data.to_csv('cv_data.csv',index=False,encoding='utf-8')


# In[68]:


df.columns


# In[69]:


df.info()


# In[70]:


df.describe()


# In[71]:


df.isnull().sum()


# In[72]:


df.duplicated().sum()


# In[73]:


# fitting simple linear regression to the training dataset
regressor = LinearRegression(normalize=True)  
regressor.fit( x_train, y_train)  


# In[74]:


# getting prediction values 
y_pred_train=regressor.predict(x_train)


# In[75]:


# loading cross validation dataset file
cv_data = pd.read_csv('cv_data.csv')
train_df.head()


# In[76]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,1].values


# In[77]:


# getting predictions on cross validation dataset
y_pred = regressor.predict(x_cv)
df = pd.DataFrame({'Actual': y_cv, 'Predicted': y_pred})  
df.head()


# In[78]:


# fileObject = open('file_Name.','wb') 


# In[79]:


# visualizing the training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Data(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[80]:


# visualizing the testing set result
plt.scatter(x_cv,y_cv,color='red')
plt.plot(x_cv,regressor.predict(x_cv),color='blue')
plt.title('Salary Data(Testing Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[81]:


# sklearn.metrics.accuracy_score(y_test,y_pred)

acc_train=sklearn.metrics.r2_score(y_train,y_pred_train)*100
print("Accuracy of train data =",acc_train)

acc_test=sklearn.metrics.r2_score(y_cv,y_pred)*100
print("Accuracy of test data =",acc_test)


# In[82]:


fileObject = open("train_data.pkl",'wb')
pickle.dump(regressor,fileObject)   
# here we close the fileObject
fileObject.close()


# In[83]:


# 


# In[ ]:




