
# coding: utf-8

# In[199]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[200]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/train.csv')


# In[201]:

train.head()


# In[202]:

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[203]:

train['Embarked'].isnull().sum()


# In[204]:

train['Cabin'].isnull().sum()


# In[205]:

train.info()


# In[206]:

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')


# In[207]:

train['Age'].mean()


# In[208]:

test['Age'].mean()


# In[209]:

train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())


# In[210]:

train = train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)


# In[211]:

train.head()


# In[212]:

train.isnull().sum()


# In[213]:

train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} )
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} )


# In[214]:

train.head()


# In[215]:

train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 1} )
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 1} )


# In[216]:

tree = DecisionTreeClassifier()


# In[217]:

y_train = train['Survived']


# In[218]:

x_train = train.copy()


# In[ ]:




# In[219]:

x_train = x_train.drop(['Survived'], axis=1)


# In[220]:

x_train.values


# In[221]:

x_train = x_train.values


# In[222]:

x_train


# In[223]:

train.info()


# In[224]:

train.head()


# In[198]:




# In[225]:

train.head()


# In[226]:

tree.fit(x_train, y_train)


# In[ ]:




# In[227]:

train.head()


# In[228]:

test.head()


# In[232]:

x_test = test.values


# In[233]:

x_test


# In[234]:

predictions = tree.predict(x_test)


# In[236]:

test.head()


# In[237]:

train.head()


# In[238]:

x_test = test.copy()


# In[241]:

x_test.drop('Survived', axis=1)


# In[243]:

x_test.head()


# In[252]:

x_test = x_test.drop('Survived', axis=1)


# In[253]:

x_test


# In[254]:

predictions = tree.predict(x_test.values)


# In[265]:

score = round(tree.score(x_train, y_train) * 100, 2)
score


# In[266]:

predictions


# In[267]:

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": predictions
    })


# In[268]:

test = pd.read_csv('data/train.csv')


# In[269]:

PassengerId = test['PassengerId']


# In[270]:

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": predictions
    })


# In[272]:

submission.info()


# In[274]:

# WAY TOO HIGH A SCORE, SCORE IS 98.650000000000006%

#POSSIBLE ISSUES:

# I DIDN'T RUN THE TEST PROPERLY ?

# OR

# I OVERFITTED THE DECISION TREES ?


# In[ ]:



