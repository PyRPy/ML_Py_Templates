
# coding: utf-8

# # Chapter 07 Preprocessing Data

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import (
    ensemble,
    model_selection,    
    preprocessing,
    tree,
)


# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.experimental import (
    enable_iterative_imputer,
)


# In[9]:


url = (
    "http://biostat.mc.vanderbilt.edu/"
    "wiki/pub/Main/DataSets/titanic3.xls"
)
df = pd.read_excel(url)


# In[10]:


X2 = pd.DataFrame(
    {
        "a": range(5),
        "b": [-100, -50, 0, 200, 1000],
    }
)
X2


# ## Standardize

# In[11]:


from sklearn import preprocessing
std = preprocessing.StandardScaler()
std.fit_transform(X2)


# In[12]:


std.scale_
std.mean_
std.var_


# In[13]:


# pandas version
X_std = (X2 - X2.mean()) / X2.std()
X_std
X_std.mean()
X_std.std()


# ## Scale to range

# In[16]:


from sklearn import preprocessing
mms = preprocessing.MinMaxScaler()
mms.fit(X2)
mms.transform(X2)


# In[17]:


(X2 - X2.min()) / (X2.max() - X2.min())


# ## Dummy variables

# In[18]:


X_cat = pd.DataFrame(
    {
        "name": ["George", "Paul"],
        "inst": ["Bass", "Guitar"],
    }
)
X_cat


# In[19]:


pd.get_dummies(X_cat, drop_first=True)


# ## Label encoder

# In[21]:


from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
lab.fit_transform(X_cat.name)


# In[22]:


lab.inverse_transform([1, 1, 0])


# In[23]:


X_cat.name.astype(
    "category"
).cat.as_ordered().cat.codes + 1


# ## Frequency encoding

# In[24]:


mapping = X_cat.name.value_counts()
X_cat.name.map(mapping)


# In[ ]:


## Pulling categories from strings


# In[25]:


from collections import Counter
c = Counter()
def triples(val):
    for i in range(len(val)):
        c[val[i : i + 3]] += 1
df.name.apply(triples)
c.most_common(10)


# In[26]:


## Regular expression


# In[27]:


df.name.str.extract(
    "([A-Za-z]+)\.", expand=False
).head()


# In[28]:


df.name.str.extract(
    "([A-Za-z]+)\.", expand=False
).value_counts()


# ## Other types of encoders

# ## Date feature engineering

# ## Add col-na feature

# In[34]:


from pandas.api.types import is_numeric_dtype
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (
            name in na_dict
        ):
            df[name + "_na"] = pd.isnull(col)
            filler = (
                na_dict[name]
                if name in na_dict
                else col.median()
            )
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
data = pd.DataFrame({"A": [0, None, 5, 100]})
fix_missing(data, data.A, "A", {})
data


# In[ ]:


## Mannaul feature engineering


# In[35]:


data = pd.DataFrame({"A": [0, None, 5, 100]})
data["A_na"] = data.A.isnull()
data["A"] = data.A.fillna(data.A.median())


# In[36]:


data


# In[37]:


agg = (
    df.groupby("cabin")
    .agg("min,max,mean,sum".split(","))
    .reset_index()
)
agg.columns = [
    "_".join(c).strip("_")
    for c in agg.columns.values
]
agg_df = df.merge(agg, on="cabin")


# In[39]:


agg_df.head()


# In[41]:


agg_df.columns


# In[42]:


df.columns


# In[43]:


# completely not proper here

