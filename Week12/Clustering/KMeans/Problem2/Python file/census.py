#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
import sklearn


# In[6]:


get_ipython().run_line_magic('time', 'df = pd.read_csv(\'Data/USCensus1990.data.txt\',delimiter=",", sep=\'\\t\', iterator=True, chunksize=10000)')
df = pd.concat(df,ignore_index=True)
df.head()


# In[7]:


# checks size of dataset
df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


# checks for null values
df.isnull().sum()


# In[11]:


# checks for duplicate values
df.duplicated().sum()


# In[12]:


# select columns
x=df.iloc[:10000,:].values


# In[13]:


# using elbow method find optimal no.of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[14]:


# visualization
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('no.of clusters')
plt.ylabel('wcss')
plt.show()


# In[15]:


# applying k-means to the mall dataset
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10)


# In[16]:


y_kmeans=kmeans.fit_predict(x)


# In[17]:


y_kmeans


# In[ ]:




