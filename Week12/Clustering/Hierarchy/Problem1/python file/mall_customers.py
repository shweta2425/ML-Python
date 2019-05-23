#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


# read file
df=pd.read_csv('Data/Mall_Customers.csv')
df.head()


# In[3]:


# checks size of dataset
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


# select columns
x=df.iloc[:,[3,4]].values


# In[9]:


denrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
# we find largest distance vertical line without crossing any horizontal line to find optimal clusters 


# In[10]:


# fit model to the dataset
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# In[11]:


y_hc = hc.fit_predict(x)
y_hc


# In[13]:


# visualizing the clusters
plt.style.use('fivethirtyeight')
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='green',label='careful 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='standard 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='red',label='target 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='orange',label='careless 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='rosybrown',label='sensible 5')
plt.title('clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[ ]:




