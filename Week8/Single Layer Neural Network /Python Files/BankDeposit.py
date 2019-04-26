#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import math
import seaborn as sb
from matplotlib import pyplot as plt


# In[31]:


# loads and read csv file
df_original=pd.read_csv("bank.csv",delimiter=";")
df =df_original
df.head()


# In[32]:


# checks data types of columns
df.dtypes


# In[33]:


# checks for null values
df.isnull().sum()


# In[34]:


df.info()


# In[35]:


df.describe()


# In[36]:


# checks for duplicate values
df.duplicated().sum()


# In[37]:


# display column names
df.columns


# In[38]:


# replacing categorical data with binary values
df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[39]:


# display shape of the dataframe
df.shape


# In[40]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[41]:


# checks correlation with all columns
print(corr['y'].sort_values(ascending=False)[:])


# In[42]:


# df.drop(['marital','contact','month'], axis=1,inplace=True)


# In[43]:


# return unique values in given column
df['y'].unique()


# In[44]:


df['y'].value_counts()


# In[45]:


sb.countplot(x='y',data=df,palette='hls')


# In[46]:


sb.boxplot(data=df)


# In[47]:


# display skewness of dataframe
target=df.skew()
sb.distplot(target)


# In[48]:


df.shape


# In[49]:


# convert categorical data into dummy (binary)  variables 
df=pd.get_dummies(df)


# In[50]:


df.boxplot(rot=45, figsize=(20,5))


# In[51]:


# scales the limit of variable
def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[52]:


df = Feature_Scaling(df)
df.head()


# In[53]:


# splits the data in 70% & 30% format
def Split(data):
    train_set=0.70*len(data)
    train=int(train_set)
#         print(train)
    test_set=0.30*len(data)
    test=int(test_set)
        
    return train,test

train,test = Split(df)


# In[54]:


train_data=df.head(train)
test_data=df.tail(test)


# In[55]:


# Separating the output(label) and the parameters(features) of dataframe
def separate(df):
    output = df.y
    return df.drop('y', axis=1), output

x_data_train,y_data_train = separate(train_data)


# In[56]:


x_data_test,y_data_test=separate(test_data)


# In[57]:


class SingleLayerNeural:
    def __init__(self):
        self.alpha = 0.221
        self.epoch = 10000
        
    def Train(self,x_train_data, y_train_data,weights,bias):
        db=0.0
        for i in range(self.epoch):
            z = (np.dot(weights.T,x_train_data.T)+bias)
            A = (1 / (1 + np.exp(-z)))
            dz = np.subtract(A,y_train_data.T)
            dw = np.dot(x_train_data.T,dz.T)/len(x_train_data)
            db = np.sum(dz,axis=1,keepdims=True)
            db = db/len(x_train_data)
            weights=np.subtract(weights,np.dot(self.alpha,dw))
            bias=np.subtract(bias,np.dot(self.alpha,db))
        return weights,bias
    
    def Test_data(self, x_test_data, weights,bias): 
        z = (np.dot(x_test_data,weights)+bias)
        A = np.array(1 / (1 + np.exp(-z)))
        # creates null array of size x_test_data (row)     
        y_prediction = np.zeros((x_test_data.shape[0], 1), dtype=float)
        
        # if output is less than or equal to 0.5 then will predict as '0' else '1'
        for i in range(0,len(A)):
            if round(A[i][0], 2) <= 0.5:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1
        return y_prediction
    
    def Accuracy(self,y_predict,y_test_data):
        count=0
        
        # if predicted and actual(label) values are same then count will get increased 
        for i in range(0,len(y_test_data)):
            if y_predict[i]==y_test_data[i]:
                count+=1
        count=(count/len(y_test_data)*100)
        return count

def main():
    # creates class object 
    obj = SingleLayerNeural()
    
    # convert data into numpy array     
    x_train_data = np.array(x_data_train)
    y_train_data = np.array(y_data_train)
    y_train_data = y_train_data.reshape(len(y_train_data),1)
        
    x_test_data = np.array(x_data_test)
    y_test_data = np.array(y_data_test)
    y_test_data = y_test_data.reshape(len(y_test_data),1)
    
    print("x_train_data",x_train_data.shape)
    print("y_train_data",y_train_data.shape) 
    print("x_test_data",x_test_data.shape)
    print("y_test_data",y_test_data.shape)  
    
    bias=1
    x_size=47
    weights = np.full((x_size+bias,1),0.1)
    
    # calling method by class object to get weights
    weights,b = obj.Train(x_train_data, y_train_data,weights,bias)
   
    # getting prediction values 
    y_predict_train = obj.Test_data(x_train_data, weights,b)
    y_predict_test = obj.Test_data(x_test_data, weights,b)
    
#     print("y_predict_train",y_predict_test)
#     for i in range(len(y_predict_test)):
#         print('> predicted=' + repr(y_predict_test[i]))
    
    # getting accuracy  
    acc_train=obj.Accuracy(y_predict_train,y_train_data)
    print("accuracy of train data=",acc_train)
    
    acc_test=obj.Accuracy(y_predict_test,y_test_data)
    print("accuracy of test data=",acc_test)
    
if __name__ == '__main__':
    main()
    


# In[ ]:




