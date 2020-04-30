#!/usr/bin/env python
# coding: utf-8

# In[120]:


from IPython.display import Image
Image (url= "https://www.readersdigest.ca/wp-content/uploads/sites/14/2012/04/titanic-facts-app-1000x675.jpg")


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pd.read_csv('train.csv')


# In[122]:


test = pd.read_csv('test.csv')


# In[7]:


train

train.isnull()
# In[8]:


train.isnull()


# In[9]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


sns.set_style('whitegrid')


# In[15]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# sns

# In[17]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[18]:


sns.countplot(x='SibSp',data=train)


# In[24]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[25]:


import cufflinks as cf


# In[26]:


cf.go_offline()


# In[28]:


train['Fare'].iplot(kind='hist',bins=50)


# In[34]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[36]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    


# In[37]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[50]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[47]:


train.drop('Cabin',axis=1)


# In[49]:


train.dropna(inplace=True)


# In[61]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[57]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[59]:


embark.head()


# In[63]:


train = pd.concat([train,sex,embark],axis=1)


# In[64]:


train.head()


# In[65]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[66]:


train.head()


# In[67]:


train.drop('PassengerId',axis=1,inplace=True)


# In[68]:


train.head()


# In[79]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[76]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[81]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pip uninstall scikit-learn


# In[ ]:


pip uninstall sklearn


# In[ ]:


pip install sklearn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[99]:


from sklearn.linear_model import LogisticRegressio


# In[101]:


logmodel = LogisticRegression()


# In[ ]:





# In[103]:


logmodel.fit(X_train,y_train) 


# In[104]:


predictions = logmodel.predict(X_test)


# In[105]:


from sklearn.metrics import classification_report


# In[106]:


print(classification_report(y_test,predictions))


# In[107]:


from sklearn.metrics import confusion_matrix


# In[108]:


confusion_matrix(y_test,predictions)


# In[ ]:





# In[ ]:





# In[ ]:




