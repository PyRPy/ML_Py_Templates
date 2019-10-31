
# coding: utf-8

# In[5]:


## Model selection
import warnings
warnings.filterwarnings('ignore')


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import (
    enable_iterative_imputer,
)
from sklearn import (
    ensemble,
    impute,
    model_selection,    
    preprocessing,
    tree,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)


# In[12]:


#from yellowbrick.model_selection import (
#    ValidationCurve,
#)


# In[3]:


url = (
    "http://biostat.mc.vanderbilt.edu/"
    "wiki/pub/Main/DataSets/titanic3.xls"
)
df = pd.read_excel(url)
df.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))
vc_viz = ValidationCurve(
    RandomForestClassifier(n_estimators=100),
    param_name="max_depth",
    param_range=np.arange(1, 11),
    cv=10,
    n_jobs=-1,
)
vc_viz.fit(X, y)
vc_viz.poof()
#fig.savefig("images/mlpr_1101.png", dpi=300)


# In[ ]:


from yellowbrick.model_selection import (
    LearningCurve,
)
fig, ax = plt.subplots(figsize=(6, 4))
lc3_viz = LearningCurve(
    RandomForestClassifier(n_estimators=100),
    cv=10,
)
lc3_viz.fit(X, y)
lc3_viz.poof()
#fig.savefig("images/mlpr_1102.png", dpi=300)


# In[ ]:


# yellowbrick loading not correctly

