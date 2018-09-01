
# coding: utf-8

# In[200]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[201]:

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


# In[202]:

train_data.head()


# In[184]:

test_data.head()


# In[185]:

train_data.info()


# In[186]:

train_data.describe()


# In[203]:

tree = DecisionTreeClassifier(random_state=0)


# In[204]:

train_data=train_data.replace({'Sex':{'male':0,'female':1}})
test_data=test_data.replace({'Sex':{'male':0,'female':1}})


# In[205]:

train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)


# In[206]:

test_data.drop('Name', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)


# In[207]:

train_data.head()


# In[208]:

train_data.drop('Embarked', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# In[209]:

test_data.drop('Embarked', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)


# In[210]:

train_data.head()


# In[211]:

test_data.head()


# In[212]:

X_train = train_data


# In[ ]:




# In[197]:

X_train.drop('Survived', axis=1, inplace=True)


# In[ ]:




# In[213]:

X_train.head()


# In[214]:

train_data


# In[215]:

y_train = train_data['Survived']


# In[173]:

y_train


# In[218]:

X_train.info()


# In[220]:

X_train.isnull().sum()


# In[222]:

age_value = X_train.Age.value_counts(normalize=True,dropna=True)


# In[223]:

age_value


# In[228]:

import numpy as np # linear algebra


# In[231]:

new_ages = np.random.choice(X_train.Age.dropna().values,size=len(X_train[X_train['Age'].isnull()]),replace=False)


# In[232]:

new_ages = pd.Series(data=new_ages,index=X_train[X_train.Age.isnull()].index)


# In[233]:

new_ages


# In[234]:

X_train['Age'].fillna(new_ages,inplace=True)


# In[235]:

X_train.info()


# In[236]:

X_train.isnull().sum()


# In[237]:

tree.fit(X_train, y_train)

tree.predict(test_data)

# In[242]:

tree.predict(test_data)


# In[ ]:



