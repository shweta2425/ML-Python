#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import random
from nltk.corpus import movie_reviews
import pickle


# In[6]:


classifier_f = open("testing.pickle", "rb")
test_data = pickle.load(classifier_f)
# classifier_f.close()


# In[7]:


classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
# classifier_f.close()


# In[8]:


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, test_data))*100)


# In[ ]:




