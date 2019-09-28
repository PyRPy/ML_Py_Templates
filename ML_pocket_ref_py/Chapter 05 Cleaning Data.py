
# coding: utf-8

# # Chapter 05 Cleaning Data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
url = (
    "http://biostat.mc.vanderbilt.edu/"
    "wiki/pub/Main/DataSets/titanic3.xls"
)
df = pd.read_excel(url)


# ## Column names

# In[2]:


# import janitor as jn
Xbad = pd.DataFrame(
    {
        "A": [1, None, 3],
        "  sales numbers ": [20.0, 30.0, None],
    }
)
# jn.clean_names(Xbad)


# In[3]:


Xbad


# In[4]:


def clean_col(name):
    return (
        name.strip().lower().replace(" ", "_")
    )

Xbad.rename(columns=clean_col)


# ## Replacing missing values

# In[5]:


Xbad.fillna(10)


# In[6]:


df.isna().any().any()

