
# coding: utf-8

# # Chapter 14 Regression

# In[13]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import (
    model_selection,
    preprocessing,
)
import warnings
warnings.filterwarnings('ignore')


# In[14]:


b = load_boston()
bos_X = pd.DataFrame(
    b.data, columns=b.feature_names
)
bos_X.head()


# ## Split data

# In[2]:


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


# ## Baseline model

# In[3]:


from sklearn.dummy import DummyRegressor
dr = DummyRegressor()
dr.fit(bos_X_train, bos_y_train)
dr.score(bos_X_test, bos_y_test)


# ## Linear Regression

# In[4]:


from sklearn.linear_model import (
    LinearRegression,
)
lr = LinearRegression()
lr.fit(bos_X_train, bos_y_train) # no scale
lr.score(bos_X_test, bos_y_test)
lr.coef_


# In[7]:


lr.score(bos_X_test, bos_y_test)


# In[5]:


lr2 = LinearRegression()
lr2.fit(bos_sX_train, bos_sy_train) # with scale
lr2.score(bos_sX_test, bos_sy_test)
lr2.intercept_
lr2.coef_


# In[6]:


lr2.score(bos_sX_test, bos_sy_test)


# In[11]:


from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
)
iris = datasets.load_iris()
iX = iris.data
iy = iris.target
lr2 = LinearRegression()
lr2.fit(iX, iy)
list(zip(iris.feature_names, lr2.coef_))


# ## SVM 

# In[15]:


from sklearn.svm import SVR
svr = SVR()
svr.fit(bos_sX_train, bos_sy_train)
svr.score(bos_sX_test, bos_sy_test)


# ## KNN

# In[16]:


from sklearn.neighbors import (
    KNeighborsRegressor,
)
knr = KNeighborsRegressor()
knr.fit(bos_sX_train, bos_sy_train)
knr.score(bos_sX_test, bos_sy_test)


# ## Decision Tree

# In[17]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(bos_X_train, bos_y_train)
dtr.score(bos_X_test, bos_y_test)


# In[18]:


import pydotplus
from io import StringIO
from sklearn.tree import export_graphviz
dot_data = StringIO()
export_graphviz(
    dtr,
    out_file=dot_data,
    feature_names=bos_X.columns,
    filled=True,
)
g = pydotplus.graph_from_dot_data(
    dot_data.getvalue()
)


# In[22]:


dot_data = StringIO()
export_graphviz(
    dtr,
    max_depth=2,
    out_file=dot_data,
    feature_names=bos_X.columns,
    filled=True,
)
g = pydotplus.graph_from_dot_data(
    dot_data.getvalue()
)


# In[24]:


for col, val in sorted(
    zip(
        bos_X.columns, dtr.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True,
)[:5]:
    print(col, val)


# ## Random Forest

# In[25]:


from sklearn.ensemble import (
    RandomForestRegressor,
)
rfr = RandomForestRegressor(
    random_state=42, n_estimators=100
)
rfr.fit(bos_X_train, bos_y_train)
rfr.score(bos_X_test, bos_y_test)


# In[30]:


for col, val in sorted(
    zip(
        bos_X.columns, rfr.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True,
)[:5]:
    print(col, val)


# ## xgboost

# In[32]:


import xgboost as xgb
xgr = xgb.XGBRegressor(random_state=42)
xgr.fit(bos_X_train, bos_y_train)
xgr.score(bos_X_test, bos_y_test)
xgr.predict(bos_X.iloc[[0]])


# In[33]:


xgr.score(bos_X_test, bos_y_test)


# In[34]:


for col, val in sorted(
    zip(
        bos_X.columns, xgr.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True,
)[:5]:
    print(col, val)


# In[35]:


fig, ax = plt.subplots(figsize=(6, 4))
xgb.plot_importance(xgr, ax=ax)
#fig.savefig("images/mlpr_1405.png", dpi=300)


# In[38]:


booster = xgr.get_booster()
print(booster.get_dump()[0])


# In[40]:


# fig, ax = plt.subplots(figsize = (6, 4))
# xgb.plot_tree(xgr, ax=ax, num_trees=0)


# ## Light GBM regression

# In[42]:


import lightgbm as lgb
lgr = lgb.LGBMRegressor(random_state=42)
lgr.fit(bos_X_train, bos_y_train)
lgr.predict(bos_X.iloc[[0]])


# In[43]:


lgr.score(bos_X_test, bos_y_test)


# In[46]:


for col, val in sorted(
    zip(
        bos_X.columns, lgr.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True,
)[:5]:
    print(col, val)


# In[47]:


fig, ax = plt.subplots(figsize=(6, 4))
lgb.plot_importance(lgr, ax=ax)
fig.tight_layout()
#fig.savefig("images/mlpr_1408.png", dpi=300)

