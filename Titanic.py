
# coding: utf-8

# In[283]:

import numpy as np
import pandas as pd
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[284]:

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[285]:

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont


# In[287]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[288]:

PassengerId = test['PassengerId']


# In[289]:

original_train = train.copy() #pass by reference


# In[290]:

full_data = [train, test]


# In[291]:

train.head(3)


# In[ ]:



