
# coding: utf-8

# In[ ]:


# Lecture 5 Linear Regression, Confidence Intervals and Standard Errors


# In[1]:


#Check Python Version
import sys
assert(sys.version_info.major==3), print(sys.version)

# Data and Stats packages
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm # RUNNING FOR ME (MSR)

# Visualization packages
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
matplotlib.rcParams['figure.figsize'] = (13.0, 6.0)

# Other Helpful fucntions
import itertools
import warnings
warnings.filterwarnings("ignore")

#Aesthetic settings
from IPython.display import display
pd.set_option('display.max_rows', 999)
pd.set_option('display.width', 500)
pd.set_option('display.notebook_repr_html', True)


# In[ ]:


# Extending Linear Regression by Transforming Predictors


# In[2]:


# Load the dataset from seaborn 
titanic = sns.load_dataset("titanic")
titanic.head()


# In[3]:


# Keep only a subset of the predictors; some are redundant, others (like deck) have too many missing values.
titanic = titanic[['age', 'sex', 'class', 'embark_town', 'alone', 'fare']]


# In[4]:


titanic.info()


# In[5]:


# Drop missing data (is this a good idea?)
titanic = titanic.dropna()
titanic.info()


# In[6]:


titanic.fare.describe()


# In[7]:


# Your code here
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
sns.distplot(titanic.fare, ax=axes[0,0])
sns.violinplot(x='fare', data=titanic, ax=axes[0,1])
sns.boxplot(x='fare', data=titanic, ax=axes[1,0])
sns.barplot(x='fare', data=titanic, ax=axes[1,1])
# Of course you should label your axes, etc.


# In[ ]:


# Exploring predictors


# In[8]:


sns.distplot(titanic.age, rug=True, rug_kws={'alpha': .1, 'color': 'k'})


# In[9]:


sns.lmplot(x="age", y="fare", hue="sex", data=titanic, size=8)


# In[10]:


titanic.sex.value_counts()


# In[11]:


titanic['class'].value_counts()


# In[12]:


titanic.class.value_counts() # wrong !


# In[13]:


sns.barplot(x="class", hue="sex", y="fare", data=titanic)


# In[14]:


sns.violinplot(x="class", hue="sex", y="fare", data=titanic)


# In[15]:


# simple linear regression on age
model1 = sm.OLS(
    titanic.fare,
    sm.add_constant(titanic['age'])
).fit()
model1.summary()


# In[16]:


# Handling categorical variables


# In[17]:


titanic_orig = titanic.copy()


# In[18]:


titanic['sex_male'] = (titanic.sex == 'male').astype(int)


# In[19]:


# Your code here
titanic['class_Second'] = (titanic['class'] == 'Second').astype(int)
titanic['class_Third'] = 1 * (titanic['class'] == 'Third') # just another way to do it


# In[20]:


titanic.head()


# In[21]:


titanic = pd.get_dummies(titanic_orig, columns=['sex', 'class'], drop_first=True)
titanic.head()


# In[22]:


# linear model
model2 = sm.OLS(
    titanic.fare,
    sm.add_constant(titanic[['age', 'sex_male', 'class_Second', 'class_Third']])
).fit()
model2.summary()


# In[ ]:


# Interactions


# In[23]:


# It seemed like gender interacted with age and class. Can we put that in our model?
titanic['sex_male_X_age'] = titanic['age'] * titanic['sex_male']
model3 = sm.OLS(
    titanic.fare,
    sm.add_constant(titanic[['age', 'sex_male', 'class_Second', 'class_Third', 'sex_male_X_age']])
).fit()
model3.summary()


# In[24]:


# It seemed like gender interacted with age and class. Can we put that in our model?
titanic['sex_male_X_class_Second'] = titanic['age'] * titanic['class_Second']
titanic['sex_male_X_class_Third'] = titanic['age'] * titanic['class_Third']
model4 = sm.OLS(
    titanic.fare,
    sm.add_constant(titanic[['age', 'sex_male', 'class_Second', 'class_Third', 
                             'sex_male_X_age', 'sex_male_X_class_Second', 'sex_male_X_class_Third']])
).fit()
model4.summary()


# In[25]:


models = [model1, model2, model3, model4]
plt.plot([model.df_model for model in models], [model.rsquared for model in models], 'x-')
plt.xlabel("Model df")
plt.ylabel("$R^2$");


# In[26]:


# Model Selection via exhaustive search selection


# In[27]:


# Import the dataset
data = pd.read_csv('dataset3.txt')
data.head()


# In[28]:


# correlation
sns.pairplot(data)


# In[29]:


data.corr()


# In[30]:


# generating heat map
sns.heatmap(data.corr())


# In[31]:


# Selecting minimal subset of predictors


# In[32]:


x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x.shape, y.shape


# In[33]:


def find_best_subset_of_size(x, y, num_predictors):
    predictors = x.columns
    
    best_r_squared = -np.inf
    best_model_data = None

    # Enumerate subsets of the given size
    subsets_of_size_k = itertools.combinations(predictors, num_predictors)

    # Inner loop: iterate through subsets_k
    for subset in subsets_of_size_k:

        # Fit regression model using ‘subset’ and calculate R^2 
        # Keep track of subset with highest R^2

        features = list(subset)
        x_subset = sm.add_constant(x[features])

        model = sm.OLS(y, x_subset).fit()
        r_squared = model.rsquared

        # Check if we get a higher R^2 value than than current max R^2.
        # If so, update our best subset 
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_model_data = {
                'r_squared': r_squared,
                'subset': features,
                'model': model
            }
    return best_model_data


# In[34]:


find_best_subset_of_size(x, y, 8)


# In[35]:


def exhaustive_search_selection(x, y):
    """Exhaustively search predictor combinations

    Parameters:
    -----------
    x : DataFrame of predictors/features
    y : response varible 
    
    
    Returns:
    -----------
    
    Dataframe of model comparisons and OLS Model with 
    lowest BIC for subset with highest R^2
    
    """
    
    predictors = x.columns
    
    stats = []
    models = dict()
    
    # Outer loop: iterate over sizes 1, 2 .... d
    for k in range(1, len(predictors)):
        
        best_size_k_model = find_best_subset_of_size(
            x, y, num_predictors=k)
        best_subset = best_size_k_model['subset']
        best_model = best_size_k_model['model']
        
        stats.append({
            'k': k,
            'formula': "y ~ {}".format(' + '.join(best_subset)),
            'bic': best_model.bic,
            'r_squared': best_model.rsquared
        })
        models[k] = best_model
        
    return pd.DataFrame(stats), models


# In[36]:


stats, models = exhaustive_search_selection(x, y)
stats


# In[37]:


stats.plot(x='k', y='bic', marker='*')


# In[38]:


stats.info()


# In[39]:


best_stat = stats.iloc[stats.bic.idxmin()]
best_stat


# In[40]:


best_k = best_stat['k']
best_bic = best_stat['bic']
best_formula = best_stat['formula']
best_r2 = best_stat['r_squared']


# In[41]:


print("The best overall model is `{formula}` with bic={bic:.2f} and R^2={r_squared:.3f}".format(
    formula=best_formula, bic=best_bic, r_squared=best_r2))


# In[42]:


models[best_k].summary()


# In[43]:


# https://harvard-iacs.github.io/2018-CS109A/sections/section-3/solutions/

