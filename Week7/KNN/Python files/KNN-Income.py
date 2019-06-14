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
df_original=pd.read_csv("classification_2.csv",delimiter=",")

df =df_original
df.head()


# In[3]:


df.columns=[
"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial_Status",
"Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
"Hours_per_week", "Country", "Target"]
df.head()


# In[4]:


df.Target.unique()


# In[5]:


# df["Target"] = df["Target"].map({ " <=50K":0, " >50K":1 })
df['Target'].replace(' <=50K',0,inplace=True)
df['Target'].replace(' >50K',1,inplace=True)


df.tail()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(keep=False,inplace=True) 
df.duplicated().sum()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.head()


# In[13]:


df.shape


# In[14]:


df.head()


# In[15]:


corr = df.corr()
sb.heatmap(corr)


# In[16]:


df = pd.get_dummies(df,columns=[
    'Workclass','Education','Martial_Status','Occupation','Relationship','Race','Sex','Country'])
df.head()


# In[17]:


df.shape


# In[18]:


corr['Target'].sort_values(ascending=False)[:]


# In[19]:


df['Target'].value_counts()


# In[20]:


sb.countplot(x='Target',data=df,palette='hls')


# In[21]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[22]:


df = Feature_Scaling(df)
# print(df)


# In[23]:


def Split(data):
    train_set=0.70*len(data)
    train=int(train_set)
#         print(train)
    test_set=0.30*len(data)
    test=int(test_set)
        
    return train,test


# In[24]:


train,test = Split(df)


# In[25]:


train_data=df.tail(train)
test_data=df.head(test)


# In[26]:


# Separating the output and the parameters data frame
def separate(df):
    output = df.Target
    return df.drop('Target', axis=1), output

x_data_train,y_data_train = separate(train_data)


# In[27]:


x_data_test,y_data_test=separate(test_data)


# In[87]:


import math
import operator
class KNN:
    def __init__(self):
        self.k=5
    
    
    def Euclidean(self,x_test_data,x_train_data,length):
        distance=0
        for i in range(length):
            distance+=pow(x_test_data[i]-x_train_data[i],2)
        return math.sqrt(distance)
    
    def get_neighbours(self,x_train_data,x_test_data,y_train_data):
        distance=[]
        length=len(x_test_data)-1
        for i in range(len(x_train_data)):
            dist=self.Euclidean(x_test_data,x_train_data[i],length)
#         print(dist)
            distance.append((y_train_data[i],dist))
            
        distance.sort(key=operator.itemgetter(1))
        neighbour=[]
        for i in range(self.k):
            neighbour.append(distance[i][0])
        return neighbour
        
    def getMajority(self,neighbors):
        majority = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in majority:
                majority[response] += 1
            else:
                majority[response] = 1
        majority = sorted(majority.items(), key=operator.itemgetter(1), reverse=True)
        return majority[0][0]
  
    
    
    def getAccuracy(self,y_test_data, predictions):
        correct = 0
#         print("Prediction",predictions)
#         print("Y test",y_test_data)
        for x in range(len(y_test_data)):
#             for i in range(len(predictions)):
            if y_test_data[x] == predictions[x]:
                correct += 1
        return (correct/float(len(y_test_data))) * 100.0

        
def main():
    obj = KNN()
    # calling method by class object
    
    x_train_data = np.array(x_data_train[:2000])
    y_train_data = np.array(y_data_train[:2000])
       
    x_test_data = np.array(x_data_test[:200])
    y_test_data = np.array(y_data_test[:200])
    predictions=[]
    for i in range(len(x_test_data)):
        neighbours = obj.get_neighbours(x_train_data,x_test_data[i],y_train_data)
        result = obj.getMajority(neighbours)
        
        predictions.append(result)
#         print('> predicted=' + repr(result) + ', actual=' + repr(y_test_data[i]))

    accuracy =obj.getAccuracy(y_test_data, predictions)
    print('\n\nAccuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()
    


# In[ ]:




