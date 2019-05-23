#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori  


# In[17]:


# read file
store_data=pd.read_csv('Data/Market_Basket_Optimisation.csv',header=None)
store_data.head()


# In[18]:


# checks size of dataset
store_data.shape


# In[23]:


#dataset is a big list and each transaction in the dataset is an inner list so we
# convert our pandas dataframe into a list of lists
records = []  
for i in range(0, 7501):  
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])


# In[25]:


association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)  
association_results = list(association_rules)  


# In[28]:


association_results[1]


# In[29]:


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


# In[ ]:




