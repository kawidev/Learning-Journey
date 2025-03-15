'''
Data: Melbourne Housing Snapshot from kaggle
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:\\Users\\szymo\\Desktop\\PythonProjects\\My learning Journey\\machine_learning\\melb_data.csv.zip')

# Preparing data 

y = df.Price
X = df.drop(['Price'], axis=1)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object' and X_train_full[col].nunique() < 10]
numerical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64', 'float64']]

#Keep selected columns only

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for i in range(1, 9):
    results[50*i] = get_score(50*i)

import matplotlib.pyplot as plt
plt.plot(list(results.keys()), list(results.values()))
plt.show()


n_estimators_best = min(results, key=results.get)
print(n_estimators_best)