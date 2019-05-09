#!/usr/bin/env python
# coding: utf-8

# In[137]:


import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import pickle
import csv
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[138]:


# read file
df_original=pd.read_csv("Position_Salaries.csv")

df =df_original
df.head()


# In[139]:


df.shape


# In[140]:


df.info()


# In[141]:


df.describe()


# In[142]:


# checks for null values
df.isnull().sum()


# In[143]:


# checks for duplicate values
df.duplicated().sum()


# In[144]:


df = df[['Level','Salary']]


# In[145]:


sc=StandardScaler()
df = sc.fit_transform(df)


# In[146]:


# splitting data into train & test dataset
train,test=train_test_split(df,test_size=0.3)


# In[147]:


# saving datasets into csv filesS
test.to_csv('test_data.csv',index=False,encoding='utf-8')
train.to_csv('train_data.csv',index=False,encoding='utf-8')


# In[148]:


# loading training data csv file
train_df = pd.read_csv('train_data.csv')
train_df.head()


# In[149]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=train_test_split(train_df,test_size=0.3)


# In[150]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,1].values


# In[151]:


sc_x = StandardScaler()
sc_y = StandardScaler()

x_train = sc_x.fit_transform(x_train.reshape(-1, 1) )
y_train = sc_y.fit_transform(y_train.reshape(-1, 1) )


# In[152]:


# saving cross validation data into csv file
cv_data.to_csv('cv_data.csv',index=False,encoding='utf-8')


# In[153]:


regressor = SVR(kernel= 'rbf')
regressor.fit(x_train,y_train)


# In[154]:


# loading cross validation dataset file
cv_data = pd.read_csv('cv_data.csv')
cv_data.head()


# In[155]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,1].values


# In[156]:


sc_x = StandardScaler()
sc_y = StandardScaler()

x_cv = sc_x.fit_transform(x_cv.reshape(-1, 1) )
y_cv = sc_y.fit_transform(y_cv.reshape(-1, 1) )


# In[158]:


class SVR:
    
    def get_predictions_train(self,x):        
        # getting prediction values
        y_pred = regressor.predict(x)
        return y_pred
    
    def get_accuracy(self,y_train,y_pred):
        Accuracy = sklearn.metrics.r2_score(y_train,y_pred)*100
        return Accuracy
        
    def visualize(self,y_pred,x,y):
        # visualizing the training set result
        x_grid=np.arange(min(x_cv),max(x_cv),0.1)
        x_grid=x_grid.reshape((len(x_grid),1))
        plt.scatter(x,y,color='red')
        plt.plot(x_grid,regressor.predict(x_grid),color='green')
        plt.title('predict salary  based on position')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
        
        
def main():
    # creates class object 
    obj = SVR()
    y_pred_train = obj.get_predictions_train(x_train)
    y_pred_test = obj.get_predictions_train(x_cv)
    
    acc_train = obj.get_accuracy(y_train,y_pred_train)
    print("Accuracy of train data =",acc_train)
    
    acc_test = obj.get_accuracy(y_cv,y_pred_test)
    print("Accuracy of test data =",acc_test)
    if acc_train >= 85 and acc_test >=  60:
        fileObject = open("train_data.pkl",'wb')
        pickle.dump(regressor,fileObject)   
        # here we close the fileObject
        fileObject.close()

     
    obj.visualize(y_pred_train,x_train,y_train)
    obj.visualize(y_pred_test,x_cv,y_cv)
    
if __name__ == '__main__':
    main()


# In[ ]:



