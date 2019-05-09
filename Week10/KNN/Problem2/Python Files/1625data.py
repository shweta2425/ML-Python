#!/usr/bin/env python
# coding: utf-8

# In[34]:


from template import Template as temp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle
from sklearn.metrics import classification_report, confusion_matrix 


# In[35]:


# read file
df=pd.read_csv('1625Data.txt',names=['octamers','flags'])
df.head()


# In[36]:


df.shape


# In[37]:


df.info()


# In[38]:


df.describe()


# In[39]:


# checks for null values
df.isnull().sum()


# In[40]:


# checks for duplicate values
df.duplicated().sum()


# In[41]:


df.replace({-1:0,1:1},inplace=True)
df.head()


# In[42]:


df.head()


# In[43]:


# Seperate all amino acids
octamers = np.array([[df["octamers"][i][j] for i in range(df.shape[0])] for j in range(8)])
print(octamers)


# In[44]:


# Store the seperated amino acids into a dataframe
df1=pd.DataFrame(octamers.T, columns=list('ABCDEFGH'))
print(df1)


# In[45]:


df=pd.concat([df1,df],axis=1)
df.drop(columns=['octamers'],inplace=True)
df.head()


# In[46]:


train,test=temp.split(df,0.2)


# In[47]:


print("train",train.shape)
print("test",test.shape)


# In[48]:


# saving datasets into csv filesS
temp.save_csv(test,'test_data.csv')
temp.save_csv(train,'train_data.csv')


# In[49]:


# loading training data csv file
train_df = temp.read_file('train_data.csv')
train_df.head()


# In[50]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=temp.split(train_df,0.2)


# In[51]:


# saving cross validation data into csv file
temp.save_csv(cv_data,'cv_data.csv')


# In[52]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,8].values


# In[53]:


x_train=pd.DataFrame(x_train)


# In[54]:


x_train=temp.oneHotEncoding(x_train)


# In[55]:


print(x_train.shape)


# In[56]:


# feature scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
# test=sc.transform(test)


# In[57]:


# fit model
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(x_train,y_train)


# In[58]:


y_pred = classifier.predict(x_train)
df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})  
df.head()


# In[59]:


# making confusion matrix
cm= confusion_matrix(y_train,y_pred)
print(cm)


# In[60]:


# loading cross validation dataset file
cv_data = temp.read_file('cv_data.csv')
cv_data.head()


# In[61]:


# separating features and labels of training dataset
x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,8].values
x_cv.shape


# In[62]:


x_cv=pd.DataFrame(x_cv)
x_cv.shape


# In[63]:


x_cv=temp.oneHotEncoding(x_cv)


# In[64]:


x_cv.shape


# In[65]:


# feature scaling
x_cv=sc.fit_transform(x_cv)


# In[66]:


class Logistic:
    
    def get_predictions(self,x):        
        # getting prediction values
        y_pred = classifier.predict(x)
        return y_pred
    
    def create_confusion_matrix(self,y,y_pred):
        # making confusion matrix
        cm= confusion_matrix(y,y_pred)
        return cm
    
    def get_accuracy(self,y_train,y_pred):
        Accuracy = sklearn.metrics.balanced_accuracy_score(y_train,y_pred)*100
        return Accuracy
        
    def visualize(self,y_pred,x,y):
        # visualizing the training set result
        
        x1,x2=np.meshgrid(np.arange(start=x[:,0].min()-1,stop=x[:,0].max()+1,step=0.01),np.arange(start=x[:,1].min()-1,stop=x[:,1].max()+1,step=0.01 ))
        plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
        plt.xlim(x1.min(),x1.max())
        plt.ylim(x2.min(),x2.max())
        
        for i,j in enumerate(np.unique(y)):
            plt.scatter(x[y==j,0],x[y==j,1],c=ListedColormap(('red','green'))(i),label=j)
        
        plt.title('predict user will click the ad or not(train dataset)')
        plt.xlabel('Age')
        plt.ylabel('estimated salary')
        plt.show()
        temp
        
def main():
    # creates class object 
    obj = Logistic()
    y_pred_train = obj.get_predictions(x_train)
    y_pred_test = obj.get_predictions(x_cv)
    
    cm_train=obj.create_confusion_matrix(y_train,y_pred_train)
    print("train matrix\n",cm_train)
    
    cm_cv=obj.create_confusion_matrix(y_cv,y_pred_test)
    print("cv matrix\n",cm_cv)
    
    acc_train = obj.get_accuracy(y_train,y_pred_train)
    print("Accuracy of train data =",acc_train)
    
    acc_test = obj.get_accuracy(y_cv,y_pred_test)
    print("Accuracy of test data =",acc_test)
    
    if acc_train >= 80 and acc_test >=  60:
        fileObject = open("train_data.pkl",'wb')
        pickle.dump(classifier,fileObject)   
        pickle.dump(sc,fileObject)
        # here we close the fileObject
        fileObject.close()

        
if __name__ == '__main__':
    main()


# In[ ]:




