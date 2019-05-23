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


get_ipython().run_line_magic('time', 'df = pd.read_csv(\'Data/USCensus1990.data.txt\',delimiter=",", sep=\'\\t\', iterator=True, chunksize=10000)')
df = pd.concat(df,ignore_index=True)
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
x=df.iloc[:10000,:].values


# In[9]:


denrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
# we find largest distance vertical line without crossing any horizontal line to find optimal clusters 


# In[10]:


# fit model to the dataset
hc = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')


# In[11]:


y_hc = hc.fit_predict(x)
y_hc


# In[ ]:





# In[ ]:




