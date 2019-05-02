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


# In[2]:


# read file
df_original=pd.read_csv("bike_sharing.csv")

df =df_original
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


# checks for null values
df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


# checks for duplicate values
df.duplicated().sum()


# In[8]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[9]:


# checks correlation with all columns
print(corr['cnt'].sort_values(ascending=False)[:])


# In[10]:


df.columns


# In[11]:


df = df[['temp','cnt']]


# In[12]:


# df.drop(columns=['holiday','dteday','instant','yr','workingday'],inplace=True)


# In[13]:


sb.boxplot(data=df)


# In[14]:


# display skewness of dataframe
target=df.skew()
sb.distplot(target)


# In[15]:


df.shape


# In[16]:


df.cnt = (np.sqrt(np.sqrt(df.cnt)))
print ('Skewness is', df.cnt.skew())
sb.distplot(df.cnt)


# In[17]:


df.shape


# In[18]:


# splitting data into train & test dataset
train,test=train_test_split(df,test_size=0.2)


# In[19]:


print("train",train.shape)
print("test",test.shape)


# In[20]:


# saving datasets into csv filesS
test.to_csv('test_data.csv',index=False,encoding='utf-8')
train.to_csv('train_data.csv',index=False,encoding='utf-8')


# In[21]:


# loading training data csv file
train_df = pd.read_csv('train_data.csv')
train_df.head()


# In[22]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=train_test_split(train_df,test_size=0.3)


# In[23]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,1].values


# In[24]:


x_train.shape


# In[25]:


# saving cross validation data into csv file
cv_data.to_csv('cv_data.csv',index=False,encoding='utf-8')


# In[26]:


# fitting simple linear regression model to the training dataset
regressor = LinearRegression(normalize=True)  
regressor.fit( x_train, y_train)  


# In[27]:


# loading cross validation dataset file
cv_data = pd.read_csv('cv_data.csv')
cv_data.head()


# In[28]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,1].values


# In[29]:


class simpleLinear:
    
    def fit_model(self,x_train,y_train):        
        # getting prediction values 
        y_pred_train=regressor.predict(x_train)
        
        return y_pred_train
    
    def ypred(self,x_cv):
        # getting predictions on cross validation dataset
        y_pred = regressor.predict(x_cv)
        return y_pred
    
    def get_accuracy(self,y_train,y_pred_train):
        error = sklearn.metrics.r2_score(y_train,y_pred_train)        
        Accuracy = (1-error)*100
        return Accuracy
        
    def visualize_trainset(self,y_pred_train):
        # visualizing the training set result
        plt.scatter(x_train,y_train,color='red')
        plt.plot(x_train,regressor.predict(x_train),color='blue')
        plt.title('Bike Data(Training Set)')
        plt.xlabel('temp')
        plt.ylabel('cnt')
        plt.show()
        
    def visualize_cvset(self,y_pred_test):
        # visualizing the testing set result
        plt.scatter(x_cv,y_cv,color='red')
        plt.plot(x_cv,regressor.predict(x_cv),color='blue')
        plt.title('bike Data(Testing Set)')
        plt.xlabel('temp')
        plt.ylabel('no.of bikes')
        plt.show()

def main():
    # creates class object 
    obj = simpleLinear()
    y_pred_train = obj.fit_model(x_train,y_train)
    y_pred_test = obj.ypred(x_cv)
    
    acc_train = obj.get_accuracy(y_train,y_pred_train)
    print("Accuracy of train data =",acc_train)
    acc_test = obj.get_accuracy(y_cv,y_pred_test)
    print("Accuracy of test data =",acc_test)
     
    obj.visualize_trainset(y_pred_train)
    obj.visualize_cvset(y_pred_test)
    
if __name__ == '__main__':
    main()
    


# In[30]:


fileObject = open("train_data.pkl",'wb')
pickle.dump(regressor,fileObject)   
# here we close the fileObject
fileObject.close()


# In[ ]:




