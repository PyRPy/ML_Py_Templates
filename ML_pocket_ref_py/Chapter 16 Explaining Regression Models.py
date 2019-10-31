
# coding: utf-8

# In[2]:


## Chapter 16 Explaining Regression Models
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn import (
    model_selection,
    preprocessing,
)


# In[5]:


b = load_boston()
bos_X = pd.DataFrame(
    b.data, columns=b.feature_names
)
bos_y = b.target
bos_X_train, bos_X_test, bos_y_train, bos_y_test = model_selection.train_test_split(
    bos_X,
    bos_y,
    test_size=0.3,
    random_state=42,
)
bos_sX = preprocessing.StandardScaler().fit_transform(
    bos_X
)
bos_sX_train, bos_sX_test, bos_sy_train, bos_sy_test = model_selection.train_test_split(
    bos_sX,
    bos_y,
    test_size=0.3,
    random_state=42,
)


# In[6]:


bos_X.head()


# In[8]:


bos_y[0:5]


# In[9]:


import xgboost as xgb
xgr = xgb.XGBRegressor(
    random_state=42, base_score=0.5
)
xgr.fit(bos_X_train, bos_y_train)


# In[10]:


sample_idx = 5
xgr.predict(bos_X.iloc[[sample_idx]])


# In[ ]:


# shall look more into statmodels

