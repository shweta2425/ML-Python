#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[42]:


# read file
df=pd.read_csv('Data/Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
df.head()


# In[43]:


df.shape


# In[44]:


nltk.download('stopwords')
corpus=[]
for i in range(0,1000):
    # cleaning the texts
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    # convert data into lowercase
    review=review.lower()
    # split words (Tokenization)
    review=review.split()
    # Stemming
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # join back the words(sentences)
    review=' '.join(review)
    corpus.append(review)


# In[45]:


for i in range(0,1000):
    print(corpus[i])


# In[46]:


# creating bag of words model
cv= CountVectorizer(max_features=1500)
x= cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values


# In[47]:


# split dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[48]:


# fit model on train data
classifier=GaussianNB()
classifier.fit(x_train,y_train)


# In[49]:


# calculate prediction values on test data
y_pred=classifier.predict(x_test)


# In[50]:


# confusion matrix of test data
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[51]:


print(classification_report(y_test,y_pred))


# In[52]:


accuracy = accuracy_score(y_pred,y_test)*100
print("accuracy of the model is ",accuracy,"%")


# In[ ]:




