#!/usr/bin/env python
# coding: utf-8

# In[38]:


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


# In[39]:


# read file
df_original=pd.read_csv("test_data.csv")

df =df_original
df.head()


# In[40]:


# separate labels and features
x_test=df.iloc[:,:-1].values
y_test=df.iloc[:,1].values


# In[41]:


# reading the pickle file

fileObject = open('train_data.pkl','rb')  
regressor = pickle.load(fileObject)  


# In[42]:


class simpleLinear:
    
    def get_predictions(self,x_test):
        
        # getting the prediction values on train model
        y_pred=regressor.predict(x_test)
        
        return y_pred
    
    def get_accuracy(self,y_pred,y_test):        

        error = sklearn.metrics.r2_score(y_test,y_pred)        
        Accuracy = (1-error)*100


        return Accuracy

    
    def visualize_cvset(self):
        # visualizing the testing set result
        plt.scatter(x_test,y_test,color='red')
        plt.plot(x_test,regressor.predict(x_test),color='blue')
        plt.title('bike Data(Testing Set)')
        plt.xlabel('temp')
        plt.ylabel('no.of bikes')
        plt.show()
    

    
def main():
    # creates class object 
    obj = simpleLinear()
    y_pred=obj.get_predictions(x_test)
    acc_test=obj.get_accuracy(y_pred,y_test)
    print("Accuracy of test data =",acc_test)
    
    obj.visualize_cvset()
if __name__ == '__main__':
    main()
    

