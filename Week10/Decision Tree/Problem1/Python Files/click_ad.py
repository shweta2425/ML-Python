
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from template import Template as temp
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier   


# In[2]:


# read file
df=temp.read_file('Social_Network_Ads.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# checks for null values
df.isnull().sum()


# In[7]:


# checks for duplicate values
df.duplicated().sum()


# In[8]:


df.columns


# In[9]:


df = df[['Age','EstimatedSalary','Purchased']]


# In[10]:


train,test=temp.split(df,0.3)


# In[11]:


print("train",train.shape)
print("test",test.shape)


# In[12]:


# # feature scaling
# sc=StandardScaler()
# train=sc.fit_transform(train)
# test=sc.transform(test)


# In[13]:


# train=pd.DataFrame(train)
# test=pd.DataFrame(test)


# In[14]:


# saving datasets into csv filesS
temp.save_csv(test,'test_data.csv')
temp.save_csv(train,'train_data.csv')


# In[15]:


# loading training data csv file
train_df = temp.read_file('train_data.csv')
train_df.head()


# In[16]:


# splitting training data into train and cross validation dataset 
train_data,cv_data=temp.split(train_df,0.3)


# In[17]:


# saving cross validation data into csv file
temp.save_csv(cv_data,'cv_data.csv')


# In[18]:


# separating features and labels of training dataset
x_train=train_data.iloc[:,[0,1]].values
y_train=train_data.iloc[:,2].values

# x_train=temp.separateFatures(train_data,[0,1])
# y_train=temp.separateFatures(train_data,2)


# In[19]:


# feature scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
# test=sc.transform(test)


# In[20]:


# fit model
classifier = DecisionTreeClassifier(criterion='entropy')  
classifier.fit(x_train,y_train)


# In[21]:


y_pred = classifier.predict(x_train)
df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})  
df.head()


# In[22]:


# making confusion matrix
cm= confusion_matrix(y_train,y_pred)
print(cm)


# In[23]:


# loading cross validation dataset file
cv_data = temp.read_file('cv_data.csv')
cv_data.head()


# In[24]:


# separate labels and features of cross validation dataset
x_cv=cv_data.iloc[:,[0,1]].values
y_cv=cv_data.iloc[:,2].values


# In[25]:


# feature scaling
x_cv=sc.fit_transform(x_cv)


# In[26]:


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
    
    if acc_train >= 70 and acc_test >=  60:
        fileObject = open("train_data.pkl",'wb')
        pickle.dump(classifier,fileObject)   
        pickle.dump(sc,fileObject)
        # here we close the fileObject
        fileObject.close()

    
#     obj.visualize(y_pred_train,x_train,y_train)
#     obj.visualize(y_pred_test,x_cv,y_cv)
    temp.visualization(y_pred_train,x_train,y_train,classifier)
    temp.visualization(y_pred,x_cv,y_cv,classifier)
    
    
    
if __name__ == '__main__':
    main()


# In[ ]:




