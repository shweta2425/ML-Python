import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import warnings
# warnings.filterwarnings('ignore')
from sklearn.metrics import *
import pickle
import csv
from sklearn.metrics import accuracy_score
import sklearn
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Template:
    
    def read_file(self,file_name):
        df=pd.read_csv(file_name)
        return df
    
    def split(self,df,size):
        # splitting data into train & test dataset
        train,test=train_test_split(df,test_size=size)
        return train,test 
     
    def save_csv(self,data,file_name):
        # saving datasets into csv filesS
        data.to_csv(file_name,index=False,encoding='utf-8')
    
    def separateFatures(self,data,col):
        # separating features and labels of training dataset
        data=data.iloc[:,col].values
        return data
    
    def visualize(self,y_pred,x,y):
        # visualizing the training set result
        plt.scatter(x,y,color='red')
        plt.plot(x,y_pred,color='blue')
        plt.title('predict salary  based on position')
        plt.xlabel('Level')
        plt.ylabel('Salary')
        plt.show()
     
    def dump_pickle(self,model_obj,file_name):
        fileObject = open(file_name,'wb')
        pickle.dump(model_obj,fileObject)     
        # here we close the fileObject
        fileObject.close()
    
    def load_pickle(self,file_name):
        # reading the pickle file
        fileObject = open(file_name,'rb')  
        model_obj = pickle.load(fileObject)
        
        return model_obj
    
    def feature_scale(self):
        # feature scaling
        sc=StandardScaler()
        train=sc.fit_transform(train)
        test=sc.transform(test)
        
        
    def visualization(self,y_pred,x,y,classifier):
        x1,x2=np.meshgrid(np.arange(start=x[:,0].min()-1,stop=x[:,0].max()+1,step=0.01),
                                             np.arange(start=x[:,1].min()-1,stop=x[:,1].max()+1,step=0.01 ))
        plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                   alpha=0.75,cmap=ListedColormap(('red','green')))
        plt.xlim(x1.min(),x1.max())
        plt.ylim(x2.min(),x2.max())
        for i,j in enumerate(np.unique(y)):
            plt.scatter(x[y==j,0],x[y==j,1],c=ListedColormap(('blue','yellow'))(i),label=j)
        
        plt.title('predict user will click the ad or not(train dataset)')
        plt.xlabel('Age')
        plt.ylabel('estimated salary')
        plt.legend()
        plt.show()
    
    def oneHotEncoding(self,x_train):
        d = defaultdict(LabelEncoder)

        # Encoding the variable
        fit = x_train.apply(lambda x: d[x.name].fit_transform(x))

        # Inverse the encoded
        fit.apply(lambda x: d[x.name].inverse_transform(x))

        # Using the dictionary to label future data
        x_train.apply(lambda x: d[x.name].transform(x))
        one_hot_encode = OneHotEncoder()
        one_hot_encode.fit(x_train)
        x_train=one_hot_encode.transform(x_train).toarray()
        return x_train  
        
        
