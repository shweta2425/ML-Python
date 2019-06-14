#!/usr/bin/env python
# coding: utf-8

# In[205]:


import tensorflow as tf 
import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import math
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
import warnings
warnings.filterwarnings('ignore')


# In[206]:


# loads and read csv file
df_original=pd.read_csv("Churn_Modelling.csv",delimiter=",")
df =df_original
df.head()


# In[207]:


df.info()


# In[208]:


df.describe()


# In[209]:


# checks data types of columns
df.dtypes


# In[210]:


# checks for null values
df.isnull().sum()


# In[211]:


# checks for duplicate values
df.duplicated().sum()


# In[212]:


# display column names
df.columns


# In[213]:


# display shape of the dataframe
df.shape


# In[214]:


# checks correlation with all columns
corr=df.corr()
sb.heatmap(corr)


# In[215]:


# checks correlation with all columns
print(corr['Exited'].sort_values(ascending=True)[:])


# In[216]:


# dropping columns whose relation is weak with label column
df.drop(['HasCrCard','Surname','CustomerId','RowNumber'],axis=1,inplace=True)
df.head()


# In[217]:


# return unique values in given column
df['Exited'].unique()


# In[218]:


df['Exited'].value_counts()


# In[219]:


sb.countplot(x='Exited',data=df,palette='hls')


# In[220]:


sb.boxplot(data=df)


# In[221]:


# display skewness of dataframe
target=df.skew()
sb.distplot(target)


# In[222]:


df.shape


# In[223]:


# convert categorical data into dummy (binary)  variables 
df=pd.get_dummies(df)


# In[224]:


# split dataset into train and test
train,test=train_test_split(df,test_size=0.3)


# In[225]:


train_data,cv_data=train_test_split(train,test_size=0.3)


# In[226]:


x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,-1:].values


# In[227]:


x_train.shape,y_train.shape


# In[228]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(x_train)


# In[229]:


x_cv=cv_data.iloc[:,:-1].values
y_cv=cv_data.iloc[:,-1:].values


# In[230]:


x_cv.shape,y_cv.shape


# In[231]:


sc.transform(x_cv)


# In[232]:


# y_train=y_train.reshape(4900,1)
# y_cv=y_cv.reshape(2100,1)


# In[233]:


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100
alpha=0.001

x = tf.placeholder('float', [None, 12])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([12, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(alpha).minimize(cost)
    
    hm_epochs = 10
    
    # create saver object
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(10):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
#                 _, c = sess.run([optimizer, cost], feed_dict={x: x_cv, y: y_cv})
                
                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        # save the variable in the disk
        saved_path = saver.save(sess, './saved_variable')
        print('model saved in {}'.format(saved_path))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#         correct1 = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_cv, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:x_train, y:y_train})*100)
        print('Accuracy:',accuracy.eval({x:x_cv, y:y_cv})*100)
        

train_neural_network(x)


# In[ ]:




