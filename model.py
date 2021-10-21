# Loading required packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pickle

# Loading data
df = pd.read_csv('emp_turn_over.csv').rename(columns={'sales': 'job_type','average_montly_hours': 'average_monthly_hours'})
# print(df.head())


# Combining support, IT and technical job types into a single type i.e. "Technical"
df['job_type'] = np.where(df['job_type'] == 'support', 'technical', df['job_type'])
df['job_type'] = np.where(df['job_type'] == 'IT', 'technical', df['job_type'])

# Splitting data to features and columns
X = df.drop(columns=['left'])
X = pd.get_dummies(X, drop_first=True)
y = df.left
# print("All Features :", X.columns)

# Feature Selection using RFE
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)
# print(rfe.support_)
selected_columns = pd.Series(index=X.columns, data=rfe.ranking_).sort_values()
selected_columns = list(selected_columns[selected_columns == 1].index)
# print("Selected Columns :", selected_columns)

# Creating a model
model = RandomForestClassifier()

model.fit(X[selected_columns].values, y.values)

pickle.dump(model, open('model.pkl', 'wb'))
