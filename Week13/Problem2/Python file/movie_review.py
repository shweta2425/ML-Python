#!/usr/bin/env python
# coding: utf-8

# In[60]:


import nltk
import random
from nltk.corpus import movie_reviews
import pickle


# In[46]:


nltk.download('movie_reviews')


# In[47]:


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)


# In[48]:


# print(documents[3])
random.shuffle(documents)


# In[49]:


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())


# In[50]:


all_words = nltk.FreqDist(all_words)
# print("top 15 most frequent words are\n",all_words.most_common(15))
# print("frequency of word stupid is",all_words["stupid"])


# In[51]:


word_features = list(all_words.keys())[:3000]


# In[52]:


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# In[53]:


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[54]:


featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[64]:


# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]
train_set = training_set[:1700]
cv_set =training_set[1700:]


# In[68]:


save_classifier = open("testing.pickle","wb")
pickle.dump(testing_set, save_classifier)
save_classifier.close()


# In[65]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[58]:


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


# In[66]:


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, cv_set))*100)


# In[67]:


classifier.show_most_informative_features(15)


# In[61]:


save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# In[62]:





# In[ ]:




