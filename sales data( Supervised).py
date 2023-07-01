#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the librraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing to dataset
df = pd.read_csv('sales.csv')


# In[3]:


pwd()


# In[4]:


df


# In[5]:


#Data Visualisation
#Building the correlation matrix
sn.heatmap(df.corr())


# In[6]:


df.shape


# In[7]:


#For checking null value
#if there is any missing value or null value
# missing = df.Administration.mean()
# df.Administration = df.Administration.fillna(missing)


# In[8]:


df.isnull().sum()


# separating dependent and independent

# In[9]:


# Here, independent vriables are Marketing spend, Adminisration, Transport, Area
# and dependent is profit


# In[10]:


x = df.drop(['Profit'],axis=1)


# In[11]:


x


# In[12]:


y = df['Profit']


# In[13]:


y


# In[14]:


#One Hot encoding


# In[15]:


city = pd.get_dummies(x["Area"],drop_first=True)


# In[16]:


#encoding tkhne krte hbe jkh data ta categorical data hbe


# In[17]:


city = pd.get_dummies(x["Area"],drop_first=True)


# In[18]:


city


# In[19]:


#drop the Area column to remove the categorical and the numerical value
x = x.drop("Area",axis=1)


# In[20]:


x


# In[21]:


#concatanation its simply adding 
x = pd.concat([x,city],axis=1)


# In[22]:


x


# In[23]:


#spliting data set in test and traing set


# In[24]:


get_ipython().system('pip install -U scikit-learn')


# In[25]:


#import library


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state=0)


# In[28]:


#fitting multiple linear regression
from sklearn.linear_model import LinearRegression


# In[29]:


regressor = LinearRegression()


# In[30]:


regressor.fit(xtrain, ytrain)


# In[31]:


y


# In[32]:


ypred = regressor.predict(xtest)


# In[33]:


ypred


# R-SQUARD VALUE

# In[34]:


from sklearn.metrics import  r2_score


# In[35]:


score=r2_score(ytest,ypred)

score
# In[36]:


#we can also have the accuracy another way
regressor.score(xtest, ytest)


# In[ ]:




