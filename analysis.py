#!/usr/bin/env python
# coding: utf-8

# 1. import all the lib file

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


# 2. check the data and read the first five lines

# In[2]:


# check the data
filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction training (with labels).csv"
data = pd.read_csv(filepath)
data.head()


# 3. get the feature of the dataset

# In[3]:


data.info()


# 4. check the situation of N/A value

# In[4]:


data.isnull().sum()


# 5. check the information of the dataset: count,mean,std,min,max.......

# In[5]:


data.describe()


# 6. show the trend of each feature

# In[6]:


data.hist(bins=50,figsize=(15,10))
plt.show


# 7. show the trend of income specially

# In[7]:


data.plot('Year of Record','Income in EUR',kind = 'scatter')


# 8. Pearson correlation coefficient is used according to the correlation between features and income.
#    at this step, I choose to drop the feature of "wear glasses"

# In[10]:


#按各个特征和income的相关性，使用皮尔逊相关系数Pearson
corr_matrix = data.corr()
print(corr_matrix['Income in EUR'].sort_values(ascending=False))


# In[ ]:




