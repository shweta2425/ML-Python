#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


# In[3]:


# read file
df_original=pd.read_csv("bike_sharing.csv")

df =df_original
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# checks for null values
df.isnull().sum()


# In[8]:


# checks for duplicate values
df.duplicated().sum()


# In[9]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[10]:


# checks correlation with all columns
print(corr['cnt'].sort_values(ascending=False)[:])


# In[11]:


df.columns


# In[12]:


df = df[['registered','cnt']]


# In[13]:


# splitting data into train & test dataset
train,test=train_test_split(df,test_size=0.2)


# In[14]:


print("train",train.shape)
print("test",test.shape)


# In[15]:


# saving datasets into csv filesS
test.to_csv('test_data.csv',index=False,encoding='utf-8')
train.to_csv('train_data.csv',index=False,encoding='utf-8')


# In[16]:


# loading training data csv file
train_df = pd.read_csv('train_data.csv')
train_df.head()


# In[17]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=train_test_split(train_df,test_size=0.4)


# In[18]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,1].values


# In[19]:


# saving cross validation data into csv file
cv_data.to_csv('cv_data.csv',index=False,encoding='utf-8')


# In[20]:


regressor=DecisionTreeRegressor()
regressor.fit(x_train,y_train)


# In[21]:


y_pred = regressor.predict(x_train)
df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})  
df.head()


# In[22]:


# loading cross validation dataset file
cv_data = pd.read_csv('cv_data.csv')
cv_data.head()


# In[23]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,1].values


# In[24]:


class DecisionTree:
    
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
        plt.plot(x_grid,regressor.predict(x_grid),color='blue')
        plt.title('predict salary  based on position')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
        
        
def main():
    # creates class object 
    obj = DecisionTree()
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




