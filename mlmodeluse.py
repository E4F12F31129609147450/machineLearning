#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# 1. read the train dataset into datatrain[]

# In[2]:


filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction training (with labels).csv"
datatrain = pd.read_csv(filepath)
datatrain.head()


# 2. read the test dataset into the datatest[]

# In[3]:


filepath="C:\\Users\\towermalta\\MLcode\\tcd ml 2019-20 income prediction test (without labels).csv"
# read_csv里面的参数是csv在你电脑上的路径
datatest = pd.read_csv(filepath)
#读取前五行数据，如果是最后五行，用data.tail()
datatest.head()


# 3. according to the analysis before, drop the faeture of "wears glasses, hair color", process the feature in both datatrain and datatest

# In[4]:


# 选取数据集中有用的特征
datatrain = datatrain.drop(labels=[ 'Wears Glasses', 'Hair Color'], axis=1)
datatrain.head()


# In[5]:


# 选取数据集中有用的特征
datatest = datatest.drop(labels=[ 'Wears Glasses', 'Hair Color'], axis=1)
datatest.head()


# 4. divide the data(datatrain and datatest) into two parts
#    first part:"gender, country, profession, university degree"
#    second part:"year of record, age, body height, size of city"

# In[6]:


data_object1 = pd.DataFrame(datatrain, columns=['Gender', 'Country', 'Profession', 'University Degree'], index=datatrain.index)
data_num1 = pd.DataFrame(datatrain, columns=['Year of Record', 'Age', 'Body Height [cm]','Size of City'], index=datatrain.index)

data_object2 = pd.DataFrame(datatest, columns=['Gender', 'Country', 'Profession', 'University Degree'], index=datatest.index)
data_num2 = pd.DataFrame(datatest, columns=['Year of Record', 'Age', 'Body Height [cm]','Size of City'], index=datatest.index)


# 5. to fill the missing value with most frequent value
#    use simpleimputer() in sklearn

# In[7]:


#填充有缺失值的行
from sklearn.impute import SimpleImputer
#SimpleImputer中输入的至少是二维矩阵  
simple = SimpleImputer(missing_values = np.nan,strategy="most_frequent")
data_object1 = simple.fit_transform(data_object1.values)
data_num1 = simple.fit_transform(data_num1.values)

data_object2 = simple.fit_transform(data_object2.values)
data_num2 = simple.fit_transform(data_num2.values)


# 6. use ordinalEncoder() to encode the object value

# In[8]:


#对object类型的数据进行编码
from sklearn import preprocessing
encoder=preprocessing.OrdinalEncoder()
data_object1=encoder.fit_transform(data_object1)
X = np.c_[data_num1,data_object1]

print(X)


# 7. got the value of target

# In[9]:


#获得y值
train_income = pd.DataFrame(datatrain, columns=['Income in EUR'], index=datatrain.index)
#y = data_conti.values
y = train_income.values
print(y)


# 8. divide the train dataset into X_train, X_test, y_train, y_test

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# 9. use minmaxscaler() to scale the feature of X_train

# In[11]:


#对特征进行缩放
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#特征很多的时候使用MinMaxScaler().partial_fit(data)来代替fit否则会报错
#scaler.fit(X)  #在这里本质是生成min(x)和max(x)
X_train = scaler.fit_transform(X_train)  #通过接口导出结果
print(X_train)


# 10. use LinearRegression to train the train dataset

# In[12]:


#训练模型
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#from sklearn.ensemble import RandomForestRegressor
#forest_reg = RandomForestRegressor()
#forest_reg.fit(X_train, y_train)


# 11. encode the X_test

# In[13]:


data_object2=encoder.fit_transform(data_object2)
X_test1 = np.c_[data_num2,data_object2]
print(X_test1)


# 12. combine the dataset of test before with the new X_test
#     this is bigger than the original X_test
#     but will perform better

# In[14]:


X_test2 = np.concatenate((X_test,X_test1),axis=0)
print(X_test2.shape)
print(X_test2)


# 13. scale the testX dataset

# In[15]:


X_test = scaler.fit_transform(X_test2)  #通过接口导出结果
print(X_test)


# 14. use the testX to predict the income 

# In[16]:


#y_pred = forest_reg.predict(X_test2)
y_pred = linreg.predict(X_test)
print(y_pred)


# In[18]:


income = pd.DataFrame(y_pred, columns=['Income'])
print(income)


# 15. put the result into a file

# In[19]:


income.to_csv('linearimpro_pred.csv') 


# In[ ]:





# In[ ]:





# In[ ]:





# # tree model

# #训练模型
# from sklearn.tree import DecisionTreeRegressor 
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(X_train,y_train)

# y_pred = tree_reg.predict(X_test)
# print(y_pred)

# # random forest model

# from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(X_train, y_train)

# y_pred = forest_reg.predict(X_test)
# print(y_pred)
