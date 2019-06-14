#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[2]:


# read file
df_original=pd.read_csv("bank.csv",delimiter=";")

df =df_original
df.head()


# In[3]:


df.dtypes


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.drop(['marital','contact','month'], axis=1,inplace=True)


# In[10]:


df.head()


# In[ ]:





# In[11]:


df.replace(['yes','no'],[1,0],inplace=True)
df.head()


# In[12]:


df = pd.get_dummies(df)
df.head()
    


# In[13]:


df.shape


# In[14]:


# sb.boxplot(data=df)
# check for ouliers
df.boxplot(rot=45, figsize=(20,5))


# In[15]:


df.shape


# In[16]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[17]:


df = Feature_Scaling(df)
df.head()


# In[18]:


def Split(data):
    train_set=0.70*len(data)
    train=int(train_set)
#         print(train)
    test_set=0.30*len(data)
    test=int(test_set)
        
    return train,test


# In[19]:


train,test = Split(df)


# In[20]:


train_data=df.head(train)
test_data=df.tail(test)


# In[21]:


# Separating the output and the parameters data frame
def separate(df):
    output = df.y
    return df.drop('y', axis=1), output

x_data_train,y_data_train = separate(train_data)


# In[22]:


x_data_test,y_data_test=separate(test_data)


# In[29]:


class Logistic_Regression:
    def __init__(self):
        # loads csv file
        self.alpha = 0.01
        self.epoch = 15000
        
    def Gradient_Descent(self, train_x_data, train_y_data,theta_vector):
#         print(theta_vector.shape,train_x_data.shape)
        for length in range(self.epoch):
            z=np.dot(theta_vector.T,train_x_data.T)
            sigmoid=(1 / (1 + np.exp(-z))) 
            a=sigmoid - train_y_data
            temp=np.dot(a,train_x_data)
            temp=np.dot(self.alpha,temp) / len(train_x_data)
            theta = theta_vector.T - temp
#         print(temp.shape,"hfdu")
        return theta    
    
    def Test_data(self, test_x_data, theta_vector):
#         print("sdhskj",test_x_data.shape,theta_vector.shape)
       
        z=np.dot(theta_vector,test_x_data.T)
        sigmoid=np.array(1 / (1 + np.exp(-z)))  
#         print(sigmoid.shape)
        y_prediction = np.zeros((test_x_data.shape[0], 1), dtype=int)
        
#         temp = np.zeros(y_predict.shape)
        for i in range(0,len(sigmoid)):
            if round(sigmoid[0][i], 2) <= 0.5:
                y_prediction[0][i] = 0
            else:
                y_prediction[0][i] = 1
        return y_prediction
    
  
    def accuracy(self, y_test_data, y_predict,y_predict_train,y_train_data):
        # accuracy
        train_acc = round(float(100 - np.mean(np.abs(y_predict_train - y_train_data)) * 100))
        test_acc = round(float(100 - np.mean(np.abs(y_predict - y_test_data)) * 100))
        print("Accuracy of train data",train_acc)
        print("Accuracy of test data",test_acc)
        
    def Accuracy(self,y_predict,y_test_data):
        count=0
#         print(y_test_data.shape,y_predict.shape)
        for i in range(0,len(y_test_data)):
            if y_predict[i]==y_test_data[i]:
                count+=1
        accuracy=(count/len(y_test_data))*100
#         print("accuracy of test data",(count/len(y_test_data))*100)
        return accuracy
    
def main():
    obj = Logistic_Regression()
    # calling method by class object
    list1 = []
    
    x_train_data = np.array(x_data_train)
    y_train_data = np.array(y_data_train)
        
    x_test_data = np.array(x_data_test)
    y_test_data = np.array(y_data_test)

    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0], 1)), x_train_data))

    x_test_data = np.column_stack((np.ones((x_test_data.shape[0], 1)), x_test_data))

    x_size = 30
    

    theta_vector = np.full((x_size+1,1),.1)

    theta_vector = obj.Gradient_Descent(x_train_data, y_train_data,theta_vector)

    y_predict_test = obj.Test_data(x_test_data, theta_vector)    
    y_predict_train = obj.Test_data(x_train_data, theta_vector)
    
    
    acc_train=obj.Accuracy(y_predict_train,y_train_data)
    print("accuracy of train data=",acc_train)
    
    acc_test=obj.Accuracy(y_predict_test,y_test_data)
    print("accuracy of test data=",acc_test) 

if __name__ == '__main__':
    main()


# In[ ]:




