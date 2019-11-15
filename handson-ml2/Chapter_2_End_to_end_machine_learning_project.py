# Chapter 2 End to end machine learning project
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import os
import tarfile
import urllib
import sys

import numpy as np
import matlibplot.pyplot as plt
import pandas as pd

# load data
housing = pd.read_csv('housing.csv')

# inspect data
housing.head()
housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

# plot some stats
housing.hist(bins = 50, figsize = (20, 15))
plt.show()

# prepare for training and test data sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

housing['income_cat'] = pd.cut(housing["median_income"], 
                           bins=[0.1, 1.5, 3.0, 4.5, 6.0, np.inf],
                           labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts()
housing["income_cat"].hist()

# unbalanced data
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)

# Discover and visualize the data to gain insights
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])

# Prepare the data for Machine Learning algorithms
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3

sample_incomplete_rows

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


housing_tr.loc[sample_incomplete_rows.index.values]

imputer.strategy
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
housing_tr.head()

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot























