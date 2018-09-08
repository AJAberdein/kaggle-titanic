
# coding: utf-8

# In[391]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[392]:

train = pd.read_csv('data/train.csv')


# In[393]:

test = pd.read_csv('data/test.csv')


# In[394]:

train.head()


# In[395]:

test.head()


# In[396]:

train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


# In[397]:

train.describe()


# In[398]:

test.describe()


# In[399]:

fig = plt.figure(figsize=(10,4))
fig.add_subplot(121)
train.Survived[train['Sex'] == 'male'].value_counts().plot(kind='pie')
fig.add_subplot(122)
train.Survived[train['Sex'] == 'female'].value_counts().plot(kind='pie')


# In[400]:

from sklearn.preprocessing import LabelEncoder
train['Sex'] = LabelEncoder().fit_transform(train['Sex']) #cool!


# In[401]:

train.head()


# In[402]:

test.head()


# In[403]:

test['Sex'] = LabelEncoder().fit_transform(test['Sex'])


# In[404]:

test.head()


# In[405]:

# titles:


# In[406]:

train['Name'].head()


# In[407]:

train['Title'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())


# In[408]:

train.head()


# In[409]:

test['Title'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())


# In[410]:

test.head()


# In[411]:

train_titles = train['Title'].unique()
test_titles = test['Title'].unique()


# In[412]:

train_titles


# In[413]:

test_titles


# In[414]:

train_means_age = dict()
test_means_age = dict()


# In[415]:

titles = list()


# In[416]:

train_titles


# In[417]:

for title in train_titles:
    titles.append(title)
    
for title in test_titles:
        titles.append(title)



# In[418]:

# titles = train_titles + test_titles


# In[419]:

titles = list(set(titles))


# In[420]:

titles


# In[421]:

mean_titles = dict()


# In[ ]:




# In[422]:

# train.Age[(train["Age"] != -1) & (train['Title'] == 'Mr')]
# train.Age[(train["Age"] == -1)]
# train[(train["Age"] == -1) & (train['Title'] == 'Mr')]


# In[423]:

train.Age[(train["Age"] != -1) & (train['Title'] == 'Mr')].mean()


# In[424]:

train['Age'].fillna(-1, inplace=True)
test['Age'].fillna(-1, inplace=True)


# In[425]:

# train['Age']


# In[426]:

#Conjoined list of titles

for title in titles:
    mean_train = train.Age[(train["Age"] != -1) & (train['Title'] == title)].mean()
    train_means_age[title] = mean_train
    
    
#     print(title)
#     print(mean_train)
#     print(train_means_age[title])
    
# print(train_means_age)


for title in titles:
    mean_test = test.Age[(test["Age"] != -1) & (test['Title'] == title)].mean()
    test_means_age[title] = mean_test
    

    
    

    
#     mean_test = test.Age[(test["Age"] != -1) & (test['Title'] == title)].mean()
#     test_means_age[title] = mean
    
    
    #get mean of train
    #get mean of test data
    
    #if mean_train && mean_test
    #mean of the two means
    
    #else if train use train
    #else test
    
    
    


# In[427]:

mean_titles = dict()

for title in titles:
    if (not np.isnan(train_means_age[title])) & (not np.isnan(test_means_age[title])):
        mean_titles[title] = (train_means_age[title] + test_means_age[title]) / 2
    elif (not np.isnan(train_means_age[title])):
        mean_titles[title] = train_means_age[title]
    elif (not np.isnan(test_means_age[title])):
        mean_titles[title] = test_means_age[title]
        
        
mean_titles

#     if (test_means_age[title]):
#         print(title)
        
    
#     print(train_means_age[title])


# In[428]:

if 'Mr' in train['Title']:
    print('hi') 
else:
    print('no')


# In[429]:

train_means_age


# In[430]:

test_means_age


# In[431]:

train.head()


# In[432]:

mean_titles[row['Name']]


# In[433]:

# for key, value in train.iterrows():
#     if row['Age'] == -1:
#         train.loc[key, 'Age'] = mean_titles[]
#         train['Age'] =  mean_titles[value['']]
#     print(value)


for key, value in train.iterrows():
#     print(value['Age'])
#     print(value['Age'] == -1)
    if value['Age'] == -1:
        print(value['Age'])
#         value['Age'] = 
#         print(mean_titles[value['Title']])
#         value['Age'] = mean_titles[value['Title']]
        train.loc[key, 'Age'] = mean_titles[value['Title']]

        print(value['Age'])
    
    


# In[434]:

for key, value in train.iterrows():
    print(value['Age'] == -1)
    if value['Age'] == -1:
        print(value['Title'])


# In[435]:

train.isnull().sum()


# In[436]:

for key, value in test.iterrows():
    if value['Age'] == -1:
        test.loc[key, 'Age'] = mean_titles[value['Title']]


# In[437]:

test.head()


# In[438]:

train.head()


# In[439]:

train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)


# In[440]:

fig = plt.figure(figsize=(15,6))

i = 1
for title in train['Title'].unique():
    fig.add_subplot(3,6,i)
    plt.title(title)
    train.Survived[train['Title'] == title].value_counts().plot(kind='pie')
    i += 1
        


# In[441]:

# fig = plt.figure(figsize=(15,6))

# i = 1
# for title in test['Title'].unique():
#     fig.add_subplot(3,6,i)
#     plt.title(title)
#     test.Survived[test['Title'] == title].value_counts().plot(kind='pie')
#     i += 1
        
    
# no survival on test


# In[442]:

mean_titles


# In[443]:

replace = {
    'Don': 0,
    'Dona': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 7,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}

train['Title'] = train['Title'].apply(lambda x: replace.get(x))
test['Title'] = test['Title'].apply(lambda x: replace.get(x))



# In[444]:

train.head()


# In[445]:

test.head()


# In[446]:

from sklearn.preprocessing import StandardScaler
train['Title'] = StandardScaler().fit_transform(train['Title'].values.reshape(-1, 1))
test['Title'] = StandardScaler().fit_transform(test['Title'].values.reshape(-1, 1))


# In[447]:

train.head()


# In[448]:

test.head()


# In[449]:

train.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)


# In[450]:

train.head()


# In[451]:

train['Age'] = StandardScaler().fit_transform(train['Age'].values.reshape(-1, 1))
test['Age'] = StandardScaler().fit_transform(test['Age'].values.reshape(-1, 1))


# In[452]:

test.isnull().sum()


# In[453]:

train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)


# In[454]:

train.head()


# In[455]:

train['Embarked'].value_counts()


# In[456]:

# import seaborn as sns

# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[457]:

train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)


# In[458]:

replace = {
    'S': 0,
    'Q': 1,
    'C': 2
}
train['Embarked'] = train['Embarked'].apply(lambda x: replace.get(x))
test['Embarked'] = test['Embarked'].apply(lambda x: replace.get(x))


# In[459]:

train['Embarked'] = StandardScaler().fit_transform(train['Embarked'].values.reshape(-1, 1))
test['Embarked'] = StandardScaler().fit_transform(test['Embarked'].values.reshape(-1, 1))


# In[460]:

train.head()


# In[461]:

test.head()


# In[462]:

test.isnull().sum()


# In[463]:

(test['Fare'].mean() + train['Fare'].mean()) / 2


# In[464]:

test['Fare'].fillna((test['Fare'].mean() + train['Fare'].mean()) / 2, inplace=True)


# In[465]:

test.isnull().sum()


# In[466]:

test.head()


# In[467]:

train_copy = train.copy()
test_copy = test.copy()

# train = train_copy.copy()


# In[468]:

train_copy.head()


# In[469]:


from sklearn.model_selection import train_test_split
survived = train['Survived']
train.drop('Survived', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train, survived, test_size=0.2, random_state=42)


# In[470]:

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

models = [
    MLPClassifier(),
    KNeighborsClassifier(),
    SVC(),
    RandomForestClassifier(n_estimators=100),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)


# In[471]:

for model in models:
    model.fit(train, survived)
    score = model.score(X_test, y_test)
    print(score)


# In[472]:

# go for the RandomForestClassifier


# In[474]:

forest = RandomForestClassifier(n_estimators=100)
forest.fit(train, survived)
y_pred = forest.predict(test.values)


# In[479]:

test_original = pd.read_csv('data/test.csv')
PassengerId = test_original['PassengerId']


# In[481]:

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })


# In[482]:

submission.to_csv('submission.csv', index=False)


# In[ ]:



