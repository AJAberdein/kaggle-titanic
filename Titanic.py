
# coding: utf-8

# In[135]:

import numpy as np
import pandas as pd
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[136]:

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[137]:

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont


# In[138]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[139]:

PassengerId = test['PassengerId']


# In[140]:

original_train = train.copy() #pass by reference


# In[141]:

full_data = [train, test]


# In[142]:

train.head(3)


# In[153]:

train['has_cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1.0)


# In[151]:

test['has_cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1.0)


# In[155]:

train


# In[156]:

# can do operations on both datasets because pass by reference
# create family size by parent/child_size + sibling_size
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[157]:

train.isnull().sum()


# In[158]:

# fill nulls with 
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[159]:

train['Fare'].median()


# In[160]:

test['Fare'].median()


# In[161]:

#if medians were different the datasets would need to be conjoined and the Fare median would be found of the complete dataset
# or is it standard practice to act as if the testing data doesn't gets shown?
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


# In[162]:

train['Age'].mean()


# In[163]:

test['Age'].mean()


# In[164]:

train['Age'].std()


# In[165]:

test['Age'].std()


# In[ ]:




# In[166]:

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


# In[167]:

# Gets the passanger name titles
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


# In[168]:

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[169]:

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[170]:

#Converting enums into ints

for dataset in full_data:
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[171]:

for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[172]:

for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;


# In[173]:

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[174]:

train['Embarked'].isnull().sum()


# In[175]:

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[176]:

train


# In[177]:

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)


# In[178]:

train.head(3)


# In[179]:

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[180]:

train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])


# In[181]:

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])


# In[182]:

title_and_sex = original_train.copy()[['Name', 'Sex']]


# In[183]:

title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)


# In[184]:

title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# In[185]:

title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])


# In[186]:

#Gini Impurity??


# In[187]:

def get_gini_impurity(survived_count, total_count):
    survival_prob = survived_count/total_count
    not_survival_prob = (1 - survival_prob)
    random_observation_survived_prob = survival_prob
    random_observation_not_survived_prob = (1 - random_observation_survived_prob)
    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob
    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob
    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob
    return gini_impurity


# In[188]:

gini_impurity_starting_node = get_gini_impurity(342, 891)
gini_impurity_starting_node


# In[189]:

gini_impurity_men = get_gini_impurity(109, 577)
gini_impurity_men


# In[190]:

gini_impurity_women = get_gini_impurity(233, 314)
gini_impurity_women


# In[191]:

men_weight = 577/891
women_weight = 314/891
weighted_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)

sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node
sex_gini_decrease


# In[192]:

gini_impurity_title_1 = get_gini_impurity(81, 517)
gini_impurity_title_1


# In[ ]:




# In[193]:

gini_impurity_title_others = get_gini_impurity(261, 374)
gini_impurity_title_others


# In[194]:

title_1_weight = 517/891
title_others_weight = 374/891
weighted_gini_impurity_title_split = (gini_impurity_title_1 * title_1_weight) + (gini_impurity_title_others * title_others_weight)

title_gini_decrease = weighted_gini_impurity_title_split - gini_impurity_starting_node
title_gini_decrease


# In[ ]:




# In[195]:

# Cross Validation Folds

cv = KFold(n_splits=10)           
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes + 1)



# In[ ]:




# In[ ]:




# In[196]:

y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values 
x_test = test.values


# In[201]:

x_train.isnull().sum()


# In[202]:

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)


# In[203]:

decision_tree.fit(x_train, y_train)


# In[204]:

y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[209]:

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(train.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
        


# In[219]:

# plot the tree graph from textbook


# In[213]:

check_call


# In[218]:

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# In[ ]:



