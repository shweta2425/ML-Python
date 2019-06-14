#!/usr/bin/env python
# coding: utf-8
 
# In[19]:


from template import Template as temp
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
import sklearn


# In[20]:


# read file
df=temp.read_file('test_data.csv')
df.head()


# In[21]:


# separating features and labels of training dataset
x_test=df.iloc[:,[0,1]].values
y_test=df.iloc[:,2].values


# In[ ]:





# In[22]:


# reading the pickle file
import pickle
fileObject = open('train_data.pkl','rb')
# sc=pickle.load(fileObject)
classifier = pickle.load(fileObject)


# In[23]:


# feature scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_test=sc.fit_transform(x_test)


# In[24]:


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
                
def main():
    # creates class object 
    obj = Logistic()
    y_pred_test = obj.get_predictions(x_test)
    
    cm=obj.create_confusion_matrix(y_test,y_pred_test)
    print("Test matrix\n",cm)
    
    acc_test = obj.get_accuracy(y_test,y_pred_test)
    print("Accuracy of test data =",acc_test)
    
    temp.visualization(y_pred_test,x_test,y_test,classifier)
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




