
# coding: utf-8

# # Chapter 04 Missing Data

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
url = (
    "http://biostat.mc.vanderbilt.edu/"
    "wiki/pub/Main/DataSets/titanic3.xls"
)
df = pd.read_excel(url)
orig_df = df


# # Examining missing data

# In[3]:


df.isnull().mean() * 100


# In[5]:


import missingno as msno
ax = msno.matrix(orig_df.sample(500))
#ax.get_figure().savefig("images/mlpr_0401.png")


# In[6]:


fig, ax = plt.subplots(figsize=(6, 4))
(1 - df.isnull().mean()).abs().plot.bar(ax=ax)


# In[7]:


ax = msno.bar(orig_df.sample(500))


# In[11]:


ax = msno.heatmap(df, figsize=(6, 6))


# In[8]:


ax = msno.dendrogram(df)


# # Dropping missing data

# In[9]:


df1 = df.dropna() # last resort


# In[10]:


df1 = df.drop(columns="cabin")


# In[11]:


df1 = df.dropna(axis=1)


# # Imputing data

# In[14]:


from sklearn.impute import SimpleImputer
num_cols = df.select_dtypes(
    include="number"
).columns
im = SimpleImputer()  # mean
imputed = im.fit_transform(df[num_cols])


# In[17]:


imputed


# In[19]:


def add_indicator(col):
    def wrapper(df):
        return df[col].isna().astype(int)

    return wrapper


# In[20]:


df1 = df.assign(
    cabin_missing=add_indicator("cabin")
)


# In[22]:


df1.tail()

