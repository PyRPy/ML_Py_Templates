
# coding: utf-8

# # Chapter 19 Pipelines

# In[1]:


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


# In[8]:


import warnings
warnings.filterwarnings("ignore")


# ## Classification pipeline

# In[2]:


from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.pipeline import Pipeline


# In[3]:


def tweak_titanic(df):
    df = df.drop(
        columns=[
            "name",
            "ticket",
            "home.dest",
            "boat",
            "body",
            "cabin",
        ]
    ).pipe(pd.get_dummies, drop_first=True)
    return df


# In[4]:


class TitanicTransformer(
    BaseEstimator, TransformerMixin
):
    def transform(self, X):
        # assumes X is output
        # from reading Excel file
        X = tweak_titanic(X)
        X = X.drop(columns="survived")
        return X
    def fit(self, X, y):
        return self


# In[5]:


pipe = Pipeline(
    [
        ("titan", TitanicTransformer()),
        ("impute", impute.IterativeImputer()),
        (
            "std",
            preprocessing.StandardScaler(),
        ),
        ("rf", RandomForestClassifier()),
    ]
)


# In[6]:


from sklearn.model_selection import (
    train_test_split,
)
url = (
    "http://biostat.mc.vanderbilt.edu/"
    "wiki/pub/Main/DataSets/titanic3.xls"
)
df = pd.read_excel(url)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df,
    df.survived,
    test_size=0.3,
    random_state=42,
)


# In[9]:


pipe.fit(X_train2, y_train2)
pipe.score(X_test2, y_test2)


# In[10]:


params = {
    "rf__max_features": [0.4, "auto"],
    "rf__n_estimators": [15, 200],
}
grid = model_selection.GridSearchCV(
    pipe, cv=3, param_grid=params
)
grid.fit(df, df.survived)


# In[11]:


grid.best_params_
pipe.set_params(**grid.best_params_)
pipe.fit(X_train2, y_train2)
pipe.score(X_test2, y_test2)


# In[12]:


from sklearn import metrics
metrics.roc_auc_score(
    y_test2, pipe.predict(X_test2)
)


# ## Regressoin pipeline

# In[13]:


from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn import (
    model_selection,
    preprocessing,
)


# In[14]:


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


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
reg_pipe = Pipeline(
    [
        (
            "std",
            preprocessing.StandardScaler(),
        ),
        ("lr", LinearRegression()),
    ]
)
reg_pipe.fit(bos_X_train, bos_y_train)
reg_pipe.score(bos_X_test, bos_y_test)


# In[16]:


reg_pipe.named_steps["lr"].intercept_
reg_pipe.named_steps["lr"].coef_


# In[17]:


from sklearn import metrics
metrics.mean_squared_error(
    bos_y_test, reg_pipe.predict(bos_X_test)
)


# In[19]:


bos_X.isna().any()


# ## PAC pipeline

# In[21]:


from sklearn.decomposition import PCA
pca_pipe = Pipeline(
    [
        ("titan", TitanicTransformer()),
        ("impute", impute.IterativeImputer()),
        (
            "std",
            preprocessing.StandardScaler(),
        ),
        ("pca", PCA()),
    ]
)
X_pca = pca_pipe.fit_transform(df, df.survived)


# In[22]:


pca_pipe.named_steps[
    "pca"
].explained_variance_ratio_
pca_pipe.named_steps["pca"].components_[0]

