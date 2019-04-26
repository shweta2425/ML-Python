#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[36]:


# read file
df_original=pd.read_csv("classification_2.csv",delimiter=",")

df =df_original
df.head()


# In[37]:


df.columns=[
"Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial_Status",
"Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
"Hours_per_week", "Country", "Target"]
df.head()


# In[38]:


df.Target.unique()


# In[39]:


# df["Target"] = df["Target"].map({ " <=50K":0, " >50K":1 })
df['Target'].replace(' <=50K',0,inplace=True)
df['Target'].replace(' >50K',1,inplace=True)


df.tail()


# In[40]:


# read file and rename columns
# df_original = pd.read_csv(
# "classification_2.csv",
# names=[
# "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial_Status",
# "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
# "Hours_per_week", "Country", "Target"])
# df =df_original
# df['newtarget'] = 0
# df.tail()


# In[41]:



# for i in range(len(df['Target'])):
#     if df['Target'][i] == ' >50K':
#         df['newtarget'][i] = 0
#     else:
#         df['newtarget'][i] = 1
# df['newtarget'].head()        


# In[42]:


df.dtypes


# In[43]:


df.isnull().sum()


# In[44]:


df.duplicated().sum()


# In[45]:


df.drop_duplicates(keep=False,inplace=True) 
df.duplicated().sum()


# In[46]:


df.info()


# In[47]:


df.describe()


# In[48]:


df.head()


# In[49]:


df.shape


# In[50]:


df.head()


# In[51]:


# df = pd.get_dummies(df,columns=['Workclass','Education','Martial_Status','Occupation','Relationship','Race','Sex','Country'])
    


# In[52]:


corr = df.corr()
sb.heatmap(corr)


# In[ ]:





# In[53]:


# get dummy variables whose are in categorical type
# for name in df.columns:
#    if df[name].dtype != "int64":
#        df[name] = pd.get_dummies(df[name]) 
# df = pd.get_dummies(df, columns=[
#     "Workclass", "Education", "Martial_Status", "Occupation", "Relationship",
#     "Race", "Sex", "Country"
# ])
# df = pd.get_dummies(df,columns=[
#     'Workclass','Education','Martial_Status','Occupation','Relationship','Race','Sex','Country'])
df = pd.get_dummies(df)
df.head()
        


# In[54]:


df.head()


# In[55]:


df.shape


# In[56]:


print(corr['Target'].sort_values(ascending=False)[:]) #top 15 values
print('----------------------')
print(corr['Target'].sort_values(ascending=False)[-5:]) #last 5 values`


# In[57]:


corr['Target'].sort_values(ascending=False)[:]


# In[58]:


df['Target'].value_counts()


# In[59]:


sb.countplot(x='Target',data=df,palette='hls')


# In[60]:


def Feature_Scaling(df):
        for column in df.columns:
            df[column] = ((df[column] - df[column].min()) /
                             (df[column].max() - df[column].min()))
        return df


# In[61]:


df = Feature_Scaling(df)
# print(df)


# In[62]:


def Split(data):
    train_set=0.70*len(data)
    train=int(train_set)
#         print(train)
    test_set=0.30*len(data)
    test=int(test_set)
        
    return train,test


# In[63]:


train,test = Split(df)


# In[64]:


train_data=df.tail(train)
test_data=df.head(test)


# In[65]:


# Separating the output and the parameters data frame
def separate(df):
    output = df.Target
    return df.drop('Target', axis=1), output

x_data_train,y_data_train = separate(train_data)


# In[66]:


x_data_test,y_data_test=separate(test_data)


# In[67]:


class Logistic_Regression:
    def __init__(self):
        # loads csv file
        self.alpha = 0.029
        self.epoch = 10550
        
    def Gradient_Descent(self, train_x_data, train_y_data,theta_vector):
#         print(theta_vector.shape,train_x_data.shape)
        for length in range(self.epoch):
            z=np.dot(theta_vector.T,train_x_data.T) 
            sigmoid=(1 / (1 + np.exp(-z))) 
            a=sigmoid - train_y_data.T  
            temp=np.dot(a,train_x_data)  
            temp=np.divide(np.dot(self.alpha,temp),len(train_x_data))
            theta = theta_vector - temp.T
        return theta
    
    def Test_data(self, test_x_data, theta_vector): 
#         print("sgd",test_x_data.shape, theta_vector.shape)
        z=np.dot(theta_vector.T,test_x_data.T)
        sigmoid=np.array(1 / (1 + np.exp(-z)))  
#         print(sigmoid.shape)
        
        y_prediction = np.zeros((test_x_data.shape[0], 1), dtype=float)
        
        for i in range(0,len(sigmoid)):
            if round(sigmoid[i][0], 2) <= 0.5:
                y_prediction[i][0] = 0
            else:
                y_prediction[i][0] = 1
        return y_prediction
    
  
            
    def Accuracy(self,y_predict,y_test_data):
        count=0
#         print(y_test_data.shape,y_predict.shape)
        for i in range(0,len(y_test_data)):
            if y_predict[i]==y_test_data[i]:
                count+=1
#         print("cnt",count)
        count=(count/len(y_test_data)*100)
        
#         print("accuracy of test data",(count/len(y_test_data))*100)
        return count
    
def main():
    obj = Logistic_Regression()
    # calling method by class object
    x_train_data = np.array(x_data_train)
    y_train_data = np.array(y_data_train)
    
    y_train_data = y_train_data.reshape(len(y_train_data),1)
 
    print("x_train_data",x_train_data.shape)
    print("y_train_data",y_train_data.shape)    
    
    x_test_data = np.array(x_data_test)
    y_test_data = np.array(y_data_test) 
    y_test_data = y_test_data.reshape(len(y_test_data),1)
    
    print("x_test_data",x_test_data.shape)
    print("y_test_data",y_test_data.shape)    
    
    x_train_data = np.column_stack((np.ones((x_train_data.shape[0], 1)), x_train_data))
    print("x_train_data",x_train_data.shape)
    
    x_test_data = np.column_stack((np.ones((x_test_data.shape[0], 1)), x_test_data))
    print("x_test_data",x_test_data.shape)

    x_size = 108
    
#     theta_vector = np.ones(((x_size + 1), 1), dtype='f')
    theta_vector = np.full((x_size+1,1),.1)
    print("theta_vector",theta_vector.shape)
    
    theta_vector = obj.Gradient_Descent(x_train_data, y_train_data,theta_vector)
    print("theta afr",theta_vector.shape)
#     print(theta_vector)

    y_predict_test = obj.Test_data(x_test_data, theta_vector)    
    y_predict_train = obj.Test_data(x_train_data, theta_vector)
    
#     print("suhgd",y_predict_train)
    
    acc_train=obj.Accuracy(y_predict_train,y_train_data)
    print("accuracy of train data=",acc_train)
    
    acc_test=obj.Accuracy(y_predict_test,y_test_data)
    print("accuracy of test data=",acc_test)
    
    
#     print(y_predict.shape)
#     acc = obj.accuracy(y_test_data, y_predict,y_predict_train,y_train_data)
if __name__ == '__main__':
    main()


# In[ ]:




