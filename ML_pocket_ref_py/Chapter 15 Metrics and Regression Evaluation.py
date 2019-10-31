
# coding: utf-8

# ## Chapter 15 Metrics and Regression Evaluation

# In[7]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn import (
    model_selection,
    preprocessing,
)
import warnings
warnings.filterwarnings('ignore')


# In[8]:


b = load_boston()
bos_X = pd.DataFrame(
    b.data, columns=b.feature_names
)
bos_y = b.target


# In[9]:


bos_X.head()


# In[10]:


bos_X_train, bos_X_test, bos_y_train, bos_y_test = model_selection.train_test_split(
    bos_X,
    bos_y,
    test_size=0.3,
    random_state=42,
)


# In[12]:


bos_sX = preprocessing.StandardScaler().fit_transform(bos_X)
bos_sX_train, bos_sX_test, bos_sy_train, bos_sy_test = model_selection.train_test_split(
    bos_sX,
    bos_y,
    test_size=0.3,
    random_state=42,
)


# In[13]:


rfr = RandomForestRegressor(
    random_state=42, n_estimators=100
)
rfr.fit(bos_X_train, bos_y_train)


# ## Metrics

# In[14]:


from sklearn import metrics
rfr.score(bos_X_test, bos_y_test)
bos_y_test_pred = rfr.predict(bos_X_test)
metrics.r2_score(bos_y_test, bos_y_test_pred)


# In[15]:


metrics.explained_variance_score(
    bos_y_test, bos_y_test_pred
)


# In[16]:


metrics.mean_absolute_error(
    bos_y_test, bos_y_test_pred
)


# In[17]:


metrics.mean_squared_error(
    bos_y_test, bos_y_test_pred
)


# In[18]:


metrics.mean_squared_log_error(
    bos_y_test, bos_y_test_pred
)


# ## Residuals plot

# In[20]:


import statsmodels.stats.api as sms
resids = bos_y_test - rfr.predict(bos_X_test)
hb = sms.het_breuschpagan(resids, bos_X_test)
labels = [
    "Lagrange multiplier statistic",
    "p-value",
    "f-value",
    "f p-value",
]
for name, num in zip(labels, hb):
    print(name, ' ', num)


# In[21]:


fig, ax = plt.subplots(figsize=(6, 4))
resids = bos_y_test - rfr.predict(bos_X_test)
pd.Series(resids, name="residuals").plot.hist(
    bins=20, ax=ax, title="Residual Histogram"
)


# In[22]:


from scipy import stats
fig, ax = plt.subplots(figsize=(6, 4))
_ = stats.probplot(resids, plot=ax)
#fig.savefig("images/mlpr_1503.png", dpi=300)


# In[23]:


stats.kstest(resids, cdf="norm")

