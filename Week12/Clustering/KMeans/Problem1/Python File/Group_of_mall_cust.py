#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
import sklearn


# In[11]:


# read file
df=pd.read_csv('Data/Mall_Customers.csv')
df.head()


# In[12]:


# checks size of dataset
df.shape


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


# checks for null values
df.isnull().sum()


# In[16]:


# checks for duplicate values
df.duplicated().sum()


# In[17]:


# select columns
x=df.iloc[:,[3,4]].values


# In[18]:


# using elbow method find optimal no.of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[19]:


# visualization
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('no.of clusters')
plt.ylabel('wcss')
plt.show()


# In[20]:


# applying k-means to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10)


# In[21]:


y_kmeans=kmeans.fit_predict(x)


# In[22]:


y_kmeans


# In[23]:


# visualizing the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='magenta',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='pink',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='cyan',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[ ]:




